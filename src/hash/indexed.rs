// Copyright © 2016  Mikhail Zabaluev <mikhail.zabaluev@gmail.com>
// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::VacantEntryState::*;

use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash, Hasher};
use std::ops::Deref;

use std::collections::hash_map::RandomState;

use zipped::LonePtr;
use super::table::{
    self,
    Bucket,
    EmptyBucket,
    FullBucket,
    RawTable,
    SafeHash
};
use super::table::BucketState::{
    Empty,
    Full,
};


pub trait HashIndexer<V> {
    type Key: ?Sized;

    fn eq_key<Q: ?Sized>(&self, val: &V, key: &Q) -> bool
        where Self::Key: Borrow<Q>, Q: Eq;

    fn hash_key<H: Hasher>(&self, val: &V, state: &mut H);
}


// The two types below below are nearly identical, except for the bound on F.
// It would be nice to be able to unify the types with
//     where F: Fn(&V) -> R, R: Borrow<K>, K: Hash

struct HashKeyFn<F> {
    func: F
}

struct HashRefKeyFn<F> {
    func: F
}

impl<F> HashKeyFn<F> {
    fn new<K, V>(func: F) -> HashKeyFn<F>
        where F: Fn(&V) -> K, K: Eq + Hash {

        HashKeyFn { func: func }
    }
}

impl<F> HashRefKeyFn<F> {
    fn new<K: ?Sized, V>(func: F) -> HashRefKeyFn<F>
        where F: Fn(&V) -> &K, K: Eq + Hash {

        HashRefKeyFn { func: func }
    }
}

impl<K, V, F> HashIndexer<V> for HashKeyFn<F>
    where F: Fn(&V) -> K, K: Hash {

    type Key = K;

    fn eq_key<Q: ?Sized>(&self, val: &V, key: &Q) -> bool
        where K: Borrow<Q>, Q: Eq {

        *(self.func)(val).borrow() == *key
    }

    fn hash_key<H: Hasher>(&self, val: &V, state: &mut H) {
        (self.func)(val).hash(state);
    }
}

impl<K: ?Sized, V, F> HashIndexer<V> for HashRefKeyFn<F>
    where F: Fn(&V) -> &K, K: Hash {

    type Key = K;

    fn eq_key<Q: ?Sized>(&self, val: &V, key: &Q) -> bool
        where K: Borrow<Q>, Q: Eq {

        *(self.func)(val).borrow() == *key
    }

    fn hash_key<H: Hasher>(&self, val: &V, state: &mut H) {
        (self.func)(val).hash(state);
    }
}

struct IdentityIndexer;

impl<T: Hash> HashIndexer<T> for IdentityIndexer {
    type Key = T;

    fn eq_key<Q: ?Sized>(&self, val: &T, key: &Q) -> bool
        where T: Borrow<Q>, Q: Eq {

        *val.borrow() == *key
    }

    fn hash_key<H: Hasher>(&self, val: &T, state: &mut H) {
        val.hash(state);
    }
}

#[test]
fn test_fn_indexer() {
    let indexer = HashRefKeyFn::new(|s: &String| &s[..]);
    let s1 = String::from("hello");
    assert!(indexer.eq_key(&s1, "hello"));
    assert!(indexer.eq_key(&s1, &s1[..]));
    assert!(!indexer.eq_key(&s1, "world"));
}

/// The default behavior of HashTable implements a load factor of 90.9%.
/// This behavior is characterized by the following condition:
///
/// - if size > 0.909 * capacity: grow the map
#[derive(Clone)]
struct DefaultResizePolicy;

impl DefaultResizePolicy {
    fn new() -> DefaultResizePolicy {
        DefaultResizePolicy
    }

    #[inline]
    fn min_capacity(&self, usable_size: usize) -> usize {
        // Here, we are rephrasing the logic by specifying the lower limit
        // on capacity:
        //
        // - if `cap < size * 1.1`: grow the map
        usable_size * 11 / 10
    }

    /// An inverse of `min_capacity`, approximately.
    #[inline]
    fn usable_capacity(&self, cap: usize) -> usize {
        // As the number of entries approaches usable capacity,
        // min_capacity(size) must be smaller than the internal capacity,
        // so that the map is not resized:
        // `min_capacity(usable_capacity(x)) <= x`.
        // The left-hand side can only be smaller due to flooring by integer
        // division.
        //
        // This doesn't have to be checked for overflow since allocation size
        // in bytes will overflow earlier than multiplication by 10.
        //
        // As per https://github.com/rust-lang/rust/pull/30991 this is updated
        // to be: (cap * den + den - 1) / num
        (cap * 10 + 10 - 1) / 11
    }
}

#[test]
fn test_resize_policy() {
    let rp = DefaultResizePolicy;
    for n in 0..1000 {
        assert!(rp.min_capacity(rp.usable_capacity(n)) <= n);
        assert!(rp.usable_capacity(rp.min_capacity(n)) <= n);
    }
}

#[derive(Clone)]
pub struct HashTable<T, F, S = RandomState> {
    // All hashes are keyed on these values, to prevent hash collision attacks.
    hash_builder: S,

    table: RawTable<LonePtr<T>>,

    indexer: F,

    resize_policy: DefaultResizePolicy,
}

/// Search for a pre-hashed key.
#[inline]
fn search_hashed<T, M, F>(table: M,
                          hash: SafeHash,
                          mut is_match: F)
                          -> InternalEntry<T, M> where
    M: Deref<Target=RawTable<LonePtr<T>>>,
    F: FnMut(&T) -> bool,
{
    // This is the only function where capacity can be zero. To avoid
    // undefined behavior when Bucket::new gets the raw bucket in this
    // case, immediately return the appropriate search result.
    if table.capacity() == 0 {
        return InternalEntry::TableIsEmpty;
    }

    let size = table.size() as isize;
    let mut probe = Bucket::new(table, hash);
    let ib = probe.index() as isize;

    loop {
        let full = match probe.peek() {
            Empty(bucket) => {
                // Found a hole!
                return InternalEntry::Vacant {
                    hash: hash,
                    elem: NoElem(bucket),
                };
            }
            Full(bucket) => bucket
        };

        let robin_ib = full.index() as isize - full.displacement() as isize;

        if ib < robin_ib {
            // Found a luckier bucket than me.
            // We can finish the search early if we hit any bucket
            // with a lower distance to initial bucket than we've probed.
            return InternalEntry::Vacant {
                hash: hash,
                elem: NeqElem(full, robin_ib as usize),
            };
        }

        // If the hash doesn't match, it can't be this one..
        if hash == full.hash() {
            // If the key doesn't match, it can't be this one..
            if is_match(full.read().into_concrete()) {
                return InternalEntry::Occupied {
                    elem: full
                };
            }
        }

        probe = full.next();
        debug_assert!(probe.index() as isize != ib + size + 1);
    }
}

impl<T, F: HashIndexer<T>> HashTable<T, F, RandomState> {
    pub fn new(indexer: F) -> HashTable<T, F, RandomState> {
        HashTable {
            hash_builder: RandomState::new(),
            resize_policy: DefaultResizePolicy::new(),
            indexer: indexer,
            table: RawTable::new(0),
        }
    }
}

impl<T, F, S> HashTable<T, F, S>
    where F: HashIndexer<T>, S: BuildHasher
{
    fn make_hash<X: ?Sized>(&self, x: &X) -> SafeHash where X: Hash {
        table::make_hash(&self.hash_builder, x)
    }

    /// Search for a key, yielding the index if it's found in the hashtable.
    /// If you already have the hash for the key lying around, use
    /// search_hashed.
    fn search<'a, Q: ?Sized>(&'a self, q: &Q)
                             -> InternalEntry<T, &'a RawTable<LonePtr<T>>>
        where F::Key: Borrow<Q>, Q: Eq + Hash
    {
        let hash = self.make_hash(q);
        search_hashed(&self.table, hash, |v| self.indexer.eq_key(v, q))
    }
}

impl<T, F, S> HashTable<T, F, S>
    where F: HashIndexer<T>, S: BuildHasher
{
    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the indexer's key type, but
    /// `Hash` and `Eq` on the borrowed form *must* match those for
    /// the key type.
    /// ```
    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<&T>
        where F::Key: Borrow<Q>, Q: Hash + Eq
    {
        self.search(k).into_occupied_bucket().map(|bucket| {
            bucket.into_refs().into_concrete()
        })
    }
}

enum InternalEntry<T, M> {
    Occupied {
        elem: FullBucket<LonePtr<T>, M>,
    },
    Vacant {
        hash: SafeHash,
        elem: VacantEntryState<T, M>,
    },
    TableIsEmpty,
}

impl<T, M> InternalEntry<T, M> {
    #[inline]
    fn into_occupied_bucket(self) -> Option<FullBucket<LonePtr<T>, M>> {
        match self {
            InternalEntry::Occupied { elem } => Some(elem),
            _ => None,
        }
    }
}

/// Possible states of a VacantEntry.
enum VacantEntryState<T, M> {
    /// The index is occupied, but the key to insert has precedence,
    /// and will kick the current one out on insertion.
    NeqElem(FullBucket<LonePtr<T>, M>, usize),
    /// The index is genuinely vacant.
    NoElem(EmptyBucket<LonePtr<T>, M>),
}
