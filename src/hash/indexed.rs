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

use self::Entry::*;
use self::VacantEntryState::*;

use std::borrow::Borrow;
use std::cmp::max;
use std::hash::{BuildHasher, Hash, Hasher};
use std::mem::{self, replace};
use std::ops::Deref;

use std::collections::hash_map::RandomState;

use zipped::LonePtr;
use super::table::{
    self,
    Bucket,
    EmptyBucket,
    FullBucket,
    FullBucketMut,
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

    fn eq_val(&self, a: &V, b: &V) -> bool;

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
    where F: Fn(&V) -> K, K: Eq + Hash {

    type Key = K;

    fn eq_key<Q: ?Sized>(&self, val: &V, key: &Q) -> bool
        where K: Borrow<Q>, Q: Eq {

        *(self.func)(val).borrow() == *key
    }

    fn eq_val(&self, a: &V, b: &V) -> bool {
        (self.func)(a) == (self.func)(b)
    }

    fn hash_key<H: Hasher>(&self, val: &V, state: &mut H) {
        (self.func)(val).hash(state);
    }
}

impl<K: ?Sized, V, F> HashIndexer<V> for HashRefKeyFn<F>
    where F: Fn(&V) -> &K, K: Eq + Hash {

    type Key = K;

    fn eq_key<Q: ?Sized>(&self, val: &V, key: &Q) -> bool
        where K: Borrow<Q>, Q: Eq {

        *(self.func)(val).borrow() == *key
    }

    fn eq_val(&self, a: &V, b: &V) -> bool {
        *(self.func)(a) == *(self.func)(b)
    }

    fn hash_key<H: Hasher>(&self, val: &V, state: &mut H) {
        (self.func)(val).hash(state);
    }
}

struct IdentityIndexer;

impl<T: Eq + Hash> HashIndexer<T> for IdentityIndexer {
    type Key = T;

    fn eq_key<Q: ?Sized>(&self, val: &T, key: &Q) -> bool
        where T: Borrow<Q>, Q: Eq {

        *val.borrow() == *key
    }

    fn eq_val(&self, a: &T, b: &T) -> bool {
        *a == *b
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

const INITIAL_LOG2_CAP: usize = 5;
const INITIAL_CAPACITY: usize = 1 << INITIAL_LOG2_CAP; // 2^5

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

/// Perform robin hood bucket stealing at the given `bucket`. You must
/// also pass the position of that bucket's initial bucket so we don't have
/// to recalculate it.
///
/// `hash` and `val` are the element to "robin hood" into the hashtable.
fn robin_hood<'a, T: 'a>(bucket: FullBucketMut<'a, LonePtr<T>>,
                         mut ib: usize,
                         mut hash: SafeHash,
                         mut val: T)
                         -> &'a mut T {
    let starting_index = bucket.index();
    let size = bucket.table().size();
    // Save the *starting point*.
    let mut bucket = bucket.stash();
    // There can be at most `size - dib` buckets to displace, because
    // in the worst case, there are `size` elements and we already are
    // `displacement` buckets away from the initial one.
    let idx_end = starting_index + size - bucket.displacement();

    loop {
        let (old_hash, old_val) = bucket.replace(hash, val);
        hash = old_hash;
        val = old_val;

        loop {
            let probe = bucket.next();
            debug_assert!(probe.index() != idx_end);

            let full_bucket = match probe.peek() {
                Empty(bucket) => {
                    // Found a hole!
                    let bucket = bucket.put(hash, val);
                    // Now that it's stolen, just read the value's pointer
                    // right out of the table! Go back to the *starting point*.
                    //
                    // This use of `into_table` is misleading. It turns the
                    // bucket, which is a FullBucket on top of a
                    // FullBucketMut, into just one FullBucketMut. The "table"
                    // refers to the inner FullBucketMut in this context.
                    return bucket.into_table().into_mut_refs().into_concrete();
                },
                Full(bucket) => bucket
            };

            let probe_ib = full_bucket.index() - full_bucket.displacement();

            bucket = full_bucket;

            // Robin hood! Steal the spot.
            if ib < probe_ib {
                ib = probe_ib;
                break;
            }
        }
    }
}

impl<T, F, S> HashTable<T, F, S>
    where F: HashIndexer<T>, S: BuildHasher
{
    fn make_hash<X: ?Sized>(&self, x: &X) -> SafeHash where X: Hash {
        let mut state = self.hash_builder.build_hasher();
        x.hash(&mut state);
        table::make_hash(&state)
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

    #[inline]
    fn search_mut<'a, Q: ?Sized>(&'a mut self, q: &Q)
                                 -> InternalEntry<T, &'a mut RawTable<LonePtr<T>>>
        where F::Key: Borrow<Q>, Q: Eq + Hash
    {
        let hash = self.make_hash(q);
        let indexer = &self.indexer;
        search_hashed(&mut self.table, hash, |v| indexer.eq_key(v, q))
    }

    // The caller should ensure that invariants by Robin Hood Hashing hold.
    fn insert_hashed_ordered(&mut self, hash: SafeHash, v: T) {
        let cap = self.table.capacity();
        let mut buckets = Bucket::new(&mut self.table, hash);
        let ib = buckets.index();

        while buckets.index() != ib + cap {
            // We don't need to compare hashes for value swap.
            // Not even DIBs for Robin Hood.
            buckets = match buckets.peek() {
                Empty(empty) => {
                    empty.put(hash, v);
                    return;
                }
                Full(b) => b.into_bucket()
            };
            buckets.next();
        }
        panic!("Internal HashMap error: Out of space.");
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
    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the `HashTable`. The collection may reserve more space to avoid
    /// frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new allocation size overflows `usize`.
    pub fn reserve(&mut self, additional: usize) {
        let new_size = self.len().checked_add(additional).expect("capacity overflow");
        let min_cap = self.resize_policy.min_capacity(new_size);

        // An invalid value shouldn't make us run out of space. This includes
        // an overflow check.
        assert!(new_size <= min_cap);

        if self.table.capacity() < min_cap {
            let new_capacity = max(min_cap.next_power_of_two(), INITIAL_CAPACITY);
            self.resize(new_capacity);
        }
    }

    /// Resizes the internal vectors to a new capacity. It's your responsibility to:
    ///   1) Make sure the new capacity is enough for all the elements, accounting
    ///      for the load factor.
    ///   2) Ensure new_capacity is a power of two or zero.
    fn resize(&mut self, new_capacity: usize) {
        assert!(self.table.size() <= new_capacity);
        assert!(new_capacity.is_power_of_two() || new_capacity == 0);

        let mut old_table = replace(&mut self.table, RawTable::new(new_capacity));
        let old_size = old_table.size();

        if old_table.capacity() == 0 || old_table.size() == 0 {
            return;
        }

        // Grow the table.
        // Specialization of the other branch.
        let mut bucket = Bucket::first(&mut old_table);

        // "So a few of the first shall be last: for many be called,
        // but few chosen."
        //
        // We'll most likely encounter a few buckets at the beginning that
        // have their initial buckets near the end of the table. They were
        // placed at the beginning as the probe wrapped around the table
        // during insertion. We must skip forward to a bucket that won't
        // get reinserted too early and won't unfairly steal others spot.
        // This eliminates the need for robin hood.
        loop {
            bucket = match bucket.peek() {
                Full(full) => {
                    if full.displacement() == 0 {
                        // This bucket occupies its ideal spot.
                        // It indicates the start of another "cluster".
                        bucket = full.into_bucket();
                        break;
                    }
                    // Leaving this bucket in the last cluster for later.
                    full.into_bucket()
                }
                Empty(b) => {
                    // Encountered a hole between clusters.
                    b.into_bucket()
                }
            };
            bucket.next();
        }

        // This is how the buckets might be laid out in memory:
        // ($ marks an initialized bucket)
        //  ________________
        // |$$$_$$$$$$_$$$$$|
        //
        // But we've skipped the entire initial cluster of buckets
        // and will continue iteration in this order:
        //  ________________
        //     |$$$$$$_$$$$$
        //                  ^ wrap around once end is reached
        //  ________________
        //  $$$_____________|
        //    ^ exit once table.size == 0
        loop {
            bucket = match bucket.peek() {
                Full(bucket) => {
                    let h = bucket.hash();
                    let (b, v) = bucket.take();
                    self.insert_hashed_ordered(h, v);
                    if b.table().size() == 0 {
                        break;
                    }
                    b.into_bucket()
                }
                Empty(b) => b.into_bucket()
            };
            bucket.next();
        }

        assert_eq!(self.table.size(), old_size);
    }

    /// Insert a pre-hashed value, without first checking
    /// that there's enough room in the buckets.
    /// Returns the newly insert value.
    ///
    /// If the equivalent indexed value is already in the table,
    /// the hashtable will be untouched and a reference to the existing
    /// element will be returned.
    fn insert_hashed_nocheck(&mut self, hash: SafeHash, v: T) -> Option<T> {
        let indexer = &self.indexer;
        let entry = search_hashed(&mut self.table, hash, |val| indexer.eq_val(&v, val)).into_entry();
        match entry {
            Some(Occupied(mut elem)) => {
                Some(elem.insert(v))
            }
            Some(Vacant(elem)) => {
                elem.insert(v);
                None
            }
            None => {
                unreachable!()
            }
        }
    }

    /// Returns the number of elements in the table.
    pub fn len(&self) -> usize { self.table.size() }

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

    /// Returns true if the table contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the indexer's key type, but
    /// `Hash` and `Eq` on the borrowed form *must* match those for
    /// the key type.
    pub fn contains_key<Q: ?Sized>(&self, k: &Q) -> bool
        where F::Key: Borrow<Q>, Q: Hash + Eq
    {
        self.search(k).into_occupied_bucket().is_some()
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the indexer's key type, but
    /// `Hash` and `Eq` on the borrowed form *must* match those for
    /// the key type.
    pub fn get_mut<Q: ?Sized>(&mut self, k: &Q) -> Option<&mut T>
        where F::Key: Borrow<Q>, Q: Hash + Eq
    {
        self.search_mut(k).into_occupied_bucket().map(|bucket| {
            bucket.into_mut_refs().into_concrete()
        })
    }

    /// Inserts a value into the table.
    ///
    /// If the map did not have a value that the indexer compared as equal
    /// to the key of this one, `None` is returned.
    ///
    /// If the map did have a duplicate value, the value is updated, and the old
    /// value is returned.
    pub fn insert(&mut self, v: T) -> Option<T> {
        let mut hash_state = self.hash_builder.build_hasher();
        self.indexer.hash_key(&v, &mut hash_state);
        let hash = table::make_hash(&hash_state);
        self.reserve(1);
        self.insert_hashed_nocheck(hash, v)
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

impl<'a, T> InternalEntry<T, &'a mut RawTable<LonePtr<T>>> {
    #[inline]
    fn into_entry(self) -> Option<Entry<'a, T>> {
        match self {
            InternalEntry::Occupied { elem } => {
                Some(Occupied(OccupiedEntry {
                    elem: elem
                }))
            }
            InternalEntry::Vacant { hash, elem } => {
                Some(Vacant(VacantEntry {
                    hash: hash,
                    elem: elem,
                }))
            }
            InternalEntry::TableIsEmpty => None
        }
    }
}

/// A view into a single location in a table, which may be vacant or occupied.
pub enum Entry<'a, T: 'a> {
    /// An occupied Entry.
    Occupied(
        OccupiedEntry<'a, T>
    ),

    /// A vacant Entry.
    Vacant(
        VacantEntry<'a, T>
    ),
}

/// A view into a single occupied location in a HashMap.
pub struct OccupiedEntry<'a, T: 'a> {
    elem: FullBucket<LonePtr<T>, &'a mut RawTable<LonePtr<T>>>,
}

/// A view into a single empty location in a HashMap.
pub struct VacantEntry<'a, T: 'a> {
    hash: SafeHash,
    elem: VacantEntryState<T, &'a mut RawTable<LonePtr<T>>>,
}

/// Possible states of a VacantEntry.
enum VacantEntryState<T, M> {
    /// The index is occupied, but the value to insert has precedence,
    /// and will kick the current one out on insertion.
    NeqElem(FullBucket<LonePtr<T>, M>, usize),
    /// The index is genuinely vacant.
    NoElem(EmptyBucket<LonePtr<T>, M>),
}

impl<'a, T> OccupiedEntry<'a, T> {
    /// Gets a reference to the value in the entry.
    pub fn get(&self) -> &T {
        self.elem.read().into_concrete()
    }

    /// Gets a mutable reference to the value in the entry.
    pub fn get_mut(&mut self) -> &mut T {
        self.elem.read_mut().into_concrete()
    }

    /// Converts the OccupiedEntry into a mutable reference to the value in the entry
    /// with a lifetime bound to the map itself
    pub fn into_mut(self) -> &'a mut T {
        self.elem.into_mut_refs().into_concrete()
    }

    /// Sets the value of the entry, and returns the entry's old value
    pub fn insert(&mut self, mut value: T) -> T {
        let old_value = self.get_mut();
        mem::swap(&mut value, old_value);
        value
    }
}

impl<'a, T: 'a> VacantEntry<'a, T> {
    /// Sets the value of the entry with the VacantEntry's key,
    /// and returns a mutable reference to it
    pub fn insert(self, value: T) -> &'a mut T {
        match self.elem {
            NeqElem(bucket, ib) => {
                robin_hood(bucket, ib, self.hash, value)
            }
            NoElem(bucket) => {
                let bucket = bucket.put(self.hash, value);
                bucket.into_mut_refs().into_concrete()
            }
        }
    }
}

#[cfg(test)]
mod test_table {
    use super::{HashTable, IdentityIndexer};

    #[test]
    fn test_insert() {
        let mut m = HashTable::new(IdentityIndexer);
        assert_eq!(m.len(), 0);
        assert!(m.insert(2).is_none());
        assert_eq!(m.len(), 1);
        assert!(m.insert(4).is_none());
        assert_eq!(m.len(), 2);
        assert_eq!(*m.get(&2).unwrap(), 2);
        assert_eq!(*m.get(&4).unwrap(), 4);
    }
}