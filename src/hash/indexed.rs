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

use std::borrow::Borrow;
use std::hash::{Hash, Hasher};

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
