// Copyright © 2016  Mikhail Zabaluev <mikhail.zabaluev@gmail.com>
// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use alloc::heap::{allocate, deallocate, EMPTY};

use std::cmp;
use std::hash::{Hash, Hasher};
use std::intrinsics::needs_drop;
use std::marker;
use std::mem::{align_of, size_of};
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr::{self, Unique};

use zipped::{ZippedPtrs, Refs, MutRefs, CloneZipped};

use self::BucketState::*;

const EMPTY_BUCKET: u64 = 0;

/// The raw hashtable, providing safe-ish access to the unzipped and highly
/// optimized arrays of hashes, keys, and values (the latter two are
/// abstracted behind an implementation of `ZippedPtrs`, to allow storing
/// just the hashed values).
///
/// This design uses less memory and is a lot faster than the naive
/// `Vec<Option<u64, K, V>>`, because we don't pay for the overhead of an
/// option on every element, and we get a generally more cache-aware design.
///
/// Essential invariants of this structure:
///
///   - if t.hashes[i] == EMPTY_BUCKET, then `Bucket::at_index(&t, i).raw`
///     points to 'undefined' contents. Don't read from it. This invariant is
///     enforced outside this module with the `EmptyBucket`, `FullBucket`,
///     and `SafeHash` types.
///
///   - An `EmptyBucket` is only constructed at an index with
///     a hash of EMPTY_BUCKET.
///
///   - A `FullBucket` is only constructed at an index with a
///     non-EMPTY_BUCKET hash.
///
///   - A `SafeHash` is only constructed for non-`EMPTY_BUCKET` hash. We get
///     around hashes of zero by changing them to 0x8000_0000_0000_0000,
///     which will likely map to the same bucket, while not being confused
///     with "empty".
///
///   - All three "arrays represented by pointers" are the same length:
///     `capacity`. This is set at creation and never changes. The arrays
///     are unzipped to save space (we don't have to pay for the padding
///     between odd sized elements, such as in a map from u64 to u8), and
///     be more cache aware (scanning through 8 hashes brings in at most
///     2 cache lines, since they're all right beside each other).
///
/// You can kind of think of this module/data structure as a safe wrapper
/// around just the "table" part of the hashtable. It enforces some
/// invariants at the type level and employs some performance trickery,
/// but in general is just a tricked out `Vec<Option<u64, K, V>>`.
#[unsafe_no_drop_flag]
pub struct RawTable<Z: ZippedPtrs> {
    capacity: usize,
    size:     usize,
    hashes:   Unique<u64>,

    // Because Z does not appear directly in any of the types in the struct,
    // inform rustc that in fact instances of Z are reachable from here.
    marker:   marker::PhantomData<Z>,
}

unsafe impl<Z> Send for RawTable<Z> where Z: ZippedPtrs, Z::Values: Send {}
unsafe impl<Z> Sync for RawTable<Z> where Z: ZippedPtrs, Z::Values: Sync {}

struct RawBucket<Z> {
    hash: *mut u64,
    data: Z,
}

impl<Z: Copy> Copy for RawBucket<Z> {}
impl<Z: Copy> Clone for RawBucket<Z> {
    fn clone(&self) -> RawBucket<Z> { *self }
}

pub struct Bucket<Z, M> {
    raw:   RawBucket<Z>,
    idx:   usize,
    table: M
}

impl<Z: Copy, M: Copy> Copy for Bucket<Z, M> {}
impl<Z: Copy, M: Copy> Clone for Bucket<Z, M> {
    fn clone(&self) -> Bucket<Z, M> { *self }
}

pub struct EmptyBucket<Z, M> {
    raw:   RawBucket<Z>,
    idx:   usize,
    table: M
}

pub struct FullBucket<Z, M> {
    raw:   RawBucket<Z>,
    idx:   usize,
    table: M
}

pub type EmptyBucketImm<'table, Z> = EmptyBucket<Z, &'table RawTable<Z>>;
pub type FullBucketImm<'table, Z> = FullBucket<Z, &'table RawTable<Z>>;

pub type EmptyBucketMut<'table, Z> = EmptyBucket<Z, &'table mut RawTable<Z>>;
pub type FullBucketMut<'table, Z> = FullBucket<Z, &'table mut RawTable<Z>>;

pub enum BucketState<Z, M> {
    Empty(EmptyBucket<Z, M>),
    Full(FullBucket<Z, M>),
}

// A GapThenFull encapsulates the state of two consecutive buckets at once.
// The first bucket, called the gap, is known to be empty.
// The second bucket is full.
pub struct GapThenFull<Z, M> {
    gap: EmptyBucket<Z, ()>,
    full: FullBucket<Z, M>,
}

/// A hash that is not zero, since we use a hash of zero to represent empty
/// buckets.
#[derive(PartialEq, Copy, Clone)]
pub struct SafeHash {
    hash: u64,
}

impl SafeHash {
    /// Peek at the hash value, which is guaranteed to be non-zero.
    #[inline(always)]
    pub fn inspect(&self) -> u64 { self.hash }
}

/// We need to remove hashes of 0. That's reserved for empty buckets.
/// This function wraps up `hash_keyed` to be the only way outside this
/// module to generate a SafeHash.
pub fn make_hash<S>(state: &S) -> SafeHash
    where S: Hasher
{
    // We need to avoid 0 in order to prevent collisions with
    // EMPTY_HASH. We can maintain our precious uniform distribution
    // of initial indexes by unconditionally setting the MSB,
    // effectively reducing 64-bits hashes to 63 bits.
    SafeHash { hash: 0x8000_0000_0000_0000 | state.finish() }
}

// `replace` casts a `*u64` to a `*SafeHash`. Since we statically
// ensure that a `FullBucket` points to an index with a non-zero hash,
// and a `SafeHash` is just a `u64` with a different name, this is
// safe.
//
// This test ensures that a `SafeHash` really IS the same size as a
// `u64`. If you need to change the size of `SafeHash` (and
// consequently made this test fail), `replace` needs to be
// modified to no longer assume this.
#[test]
fn can_alias_safehash_as_u64() {
    assert_eq!(size_of::<SafeHash>(), size_of::<u64>())
}

impl<Z: ZippedPtrs> RawBucket<Z> {
    unsafe fn offset(&self, count: isize) -> RawBucket<Z> {
        RawBucket {
            hash: self.hash.offset(count),
            data: self.data.offset(count)
        }
    }
}

// Buckets hold references to the table.
impl<Z, M> FullBucket<Z, M> {
    /// Borrow a reference to the table.
    pub fn table(&self) -> &M {
        &self.table
    }
    /// Move out the reference to the table.
    pub fn into_table(self) -> M {
        self.table
    }
    /// Get the raw index.
    pub fn index(&self) -> usize {
        self.idx
    }
}

impl<Z, M> EmptyBucket<Z, M> {
    /// Borrow a reference to the table.
    pub fn table(&self) -> &M {
        &self.table
    }
}

impl<Z, M> Bucket<Z, M> {
    /// Get the raw index.
    pub fn index(&self) -> usize {
        self.idx
    }
}

impl<Z, M> Deref for FullBucket<Z, M>
    where Z: ZippedPtrs, M: Deref<Target=RawTable<Z>>
{
    type Target = RawTable<Z>;
    fn deref(&self) -> &RawTable<Z> {
        &self.table
    }
}

/// `Put` is implemented for types which provide access to a table and cannot be invalidated
///  by filling a bucket. A similar implementation for `Take` is possible.
pub trait Put<Z: ZippedPtrs> {
    unsafe fn borrow_table_mut(&mut self) -> &mut RawTable<Z>;
}


impl<'t, Z> Put<Z> for &'t mut RawTable<Z> where Z: ZippedPtrs {
    unsafe fn borrow_table_mut(&mut self) -> &mut RawTable<Z> {
        *self
    }
}

impl<Z, M> Put<Z> for Bucket<Z, M> where Z: ZippedPtrs, M: Put<Z> {
    unsafe fn borrow_table_mut(&mut self) -> &mut RawTable<Z> {
        self.table.borrow_table_mut()
    }
}

impl<Z, M> Put<Z> for FullBucket<Z, M> where Z: ZippedPtrs, M: Put<Z> {
    unsafe fn borrow_table_mut(&mut self) -> &mut RawTable<Z> {
        self.table.borrow_table_mut()
    }
}

impl<Z, M> Bucket<Z, M>
    where Z: ZippedPtrs, M: Deref<Target=RawTable<Z>>
{
    pub fn new(table: M, hash: SafeHash) -> Bucket<Z, M> {
        Bucket::at_index(table, hash.inspect() as usize)
    }

    pub fn at_index(table: M, ib_index: usize) -> Bucket<Z, M> {
        // if capacity is 0, then the RawBucket will be populated with bogus pointers.
        // This is an uncommon case though, so avoid it in release builds.
        debug_assert!(table.capacity() > 0, "Table should have capacity at this point");
        let ib_index = ib_index & (table.capacity() - 1);
        Bucket {
            raw: unsafe {
               table.first_bucket_raw().offset(ib_index as isize)
            },
            idx: ib_index,
            table: table
        }
    }

    pub fn first(table: M) -> Bucket<Z, M> {
        Bucket {
            raw: table.first_bucket_raw(),
            idx: 0,
            table: table
        }
    }

    /// Reads a bucket at a given index, returning an enum indicating whether
    /// it's initialized or not. You need to match on this enum to get
    /// the appropriate types to call most of the other functions in
    /// this module.
    pub fn peek(self) -> BucketState<Z, M> {
        match unsafe { *self.raw.hash } {
            EMPTY_BUCKET =>
                Empty(EmptyBucket {
                    raw: self.raw,
                    idx: self.idx,
                    table: self.table
                }),
            _ =>
                Full(FullBucket {
                    raw: self.raw,
                    idx: self.idx,
                    table: self.table
                })
        }
    }

    /// Modifies the bucket pointer in place to make it point to the next slot.
    pub fn next(&mut self) {
        self.idx += 1;
        let range = self.table.capacity();
        // This code is branchless thanks to a conditional move.
        let dist = if self.idx & (range - 1) == 0 {
            1 - range as isize
        } else {
            1
        };
        unsafe {
            self.raw = self.raw.offset(dist);
        }
    }
}

impl<Z, M> EmptyBucket<Z, M>
    where Z: ZippedPtrs + Copy, M: Deref<Target=RawTable<Z>>
{
    #[inline]
    pub fn next(self) -> Bucket<Z, M> {
        let mut bucket = self.into_bucket();
        bucket.next();
        bucket
    }

    #[inline]
    pub fn into_bucket(self) -> Bucket<Z, M> {
        Bucket {
            raw: self.raw,
            idx: self.idx,
            table: self.table
        }
    }

    pub fn gap_peek(self) -> Option<GapThenFull<Z, M>> {
        let gap = EmptyBucket {
            raw: self.raw,
            idx: self.idx,
            table: ()
        };

        match self.next().peek() {
            Full(bucket) => {
                Some(GapThenFull {
                    gap: gap,
                    full: bucket
                })
            }
            Empty(..) => None
        }
    }
}

impl<Z, M> EmptyBucket<Z, M> where Z: ZippedPtrs, M: Put<Z>
{
    /// Puts given table entry, along with the key's hash,
    /// into this bucket in the hashtable. Note how `self` is 'moved' into
    /// this function, because this slot will no longer be empty when
    /// we return! A `FullBucket` is returned for later use, pointing to
    /// the newly-filled slot in the hashtable.
    ///
    /// Use `make_hash` to construct a `SafeHash` to pass to this function.
    pub fn put(mut self, hash: SafeHash, data: Z::Values)
               -> FullBucket<Z, M> {
        unsafe {
            *self.raw.hash = hash.inspect();
            self.raw.data.write(data);

            self.table.borrow_table_mut().size += 1;
        }

        FullBucket { raw: self.raw, idx: self.idx, table: self.table }
    }
}

impl<Z, M> FullBucket<Z, M>
    where Z: ZippedPtrs + Copy, M: Deref<Target=RawTable<Z>>
{
    #[inline]
    pub fn next(self) -> Bucket<Z, M> {
        let mut bucket = self.into_bucket();
        bucket.next();
        bucket
    }

    #[inline]
    pub fn into_bucket(self) -> Bucket<Z, M> {
        Bucket {
            raw: self.raw,
            idx: self.idx,
            table: self.table
        }
    }

    /// Duplicates the current position. This can be useful for operations
    /// on two or more buckets.
    pub fn stash(self) -> FullBucket<Z, Self> {
        FullBucket {
            raw: self.raw,
            idx: self.idx,
            table: self,
        }
    }

    /// Get the distance between this bucket and the 'ideal' location
    /// as determined by the key's hash stored in it.
    ///
    /// In the cited blog posts above, this is called the "distance to
    /// initial bucket", or DIB. Also known as "probe count".
    pub fn displacement(&self) -> usize {
        // Calculates the distance one has to travel when going from
        // `hash mod capacity` onwards to `idx mod capacity`, wrapping around
        // if the destination is not reached before the end of the table.
        (self.idx.wrapping_sub(self.hash().inspect() as usize)) & (self.table.capacity() - 1)
    }

    #[inline]
    pub fn hash(&self) -> SafeHash {
        unsafe {
            SafeHash {
                hash: *self.raw.hash
            }
        }
    }
}

impl<Z, M> FullBucket<Z, M>
    where Z: ZippedPtrs + Copy, M: Deref<Target=RawTable<Z>>
{
    /// Gets references to the entry values at a given index.
    pub fn read<'t>(&'t self) -> Refs<'t, Z> {
        unsafe {
            Refs::from_ptrs(self.raw.data)
        }
    }
}

impl<Z, M> FullBucket<Z, M>
    where Z: CloneZipped
{
    pub fn clone_values(&self) -> Z::Values {
        unsafe {
            self.raw.data.clone_zipped()
        }
    }
}

// We take a mutable reference to the table instead of accepting anything that
// implements `DerefMut` to prevent fn `take` from being called on `stash`ed
// buckets.
impl<'t, Z: ZippedPtrs + Copy> FullBucket<Z, &'t mut RawTable<Z>> {
    /// Removes this bucket's key and value from the hashtable.
    ///
    /// This works similarly to `put`, building an `EmptyBucket` out of the
    /// taken bucket.
    pub fn take(mut self) -> (EmptyBucket<Z, &'t mut RawTable<Z>>, Z::Values) {
        self.table.size -= 1;

        unsafe {
            *self.raw.hash = EMPTY_BUCKET;
            (
                EmptyBucket {
                    raw: self.raw,
                    idx: self.idx,
                    table: self.table
                },
                self.raw.data.read()
            )
        }
    }
}

// This use of `Put` is misleading and restrictive, but safe and sufficient for our use cases
// where `M` is a full bucket or table reference type with mutable access to the table.
impl<Z, M> FullBucket<Z, M> where Z: ZippedPtrs, M: Put<Z> {
    pub fn replace(&mut self, h: SafeHash, v: Z::Values) -> (SafeHash, Z::Values) {
        unsafe {
            let old_hash = ptr::replace(self.raw.hash as *mut SafeHash, h);
            let old_val  = self.raw.data.replace(v);

            (old_hash, old_val)
        }
    }
}

impl<Z, M> FullBucket<Z, M>
    where Z: ZippedPtrs + Copy, M: Deref<Target=RawTable<Z>> + DerefMut
{
    /// Gets mutable references to the key and value at a given index.
    pub fn read_mut<'t>(&'t mut self) -> MutRefs<'t, Z> {
        unsafe {
            MutRefs::from_ptrs(self.raw.data)
        }
    }
}

impl<'t, Z, M> FullBucket<Z, M>
    where Z: ZippedPtrs, M: Deref<Target=RawTable<Z>> + 't
{
    /// Exchange a bucket state for immutable references into the table.
    /// Because the underlying reference to the table is also consumed,
    /// no further changes to the structure of the table are possible;
    /// in exchange for this, the returned references have a longer lifetime
    /// than the references returned by `read()`.
    pub fn into_refs(self) -> Refs<'t, Z> {
        unsafe {
            Refs::from_ptrs(self.raw.data)
        }
    }
}

impl<'t, Z, M> FullBucket<Z, M>
    where Z: ZippedPtrs, M: Deref<Target=RawTable<Z>> + DerefMut + 't
{
    /// This works similarly to `into_refs`, exchanging a bucket state
    /// for mutable references into the table.
    pub fn into_mut_refs(self) -> MutRefs<'t, Z> {
        unsafe {
            MutRefs::from_ptrs(self.raw.data)
        }
    }
}

impl<Z, M> GapThenFull<Z, M> {
    #[inline]
    pub fn full(&self) -> &FullBucket<Z, M> {
        &self.full
    }
}

impl<Z, M> GapThenFull<Z, M>
    where Z: ZippedPtrs + Copy, M: Deref<Target=RawTable<Z>>
{
    pub fn shift(mut self) -> Option<GapThenFull<Z, M>> {
        unsafe {
            *self.gap.raw.hash = mem::replace(&mut *self.full.raw.hash, EMPTY_BUCKET);
            ZippedPtrs::copy_nonoverlapping(&self.full.raw.data,
                                            &self.gap.raw.data,
                                            1);
        }

        let FullBucket { raw: prev_raw, idx: prev_idx, .. } = self.full;

        match self.full.next().peek() {
            Full(bucket) => {
                self.gap.raw = prev_raw;
                self.gap.idx = prev_idx;

                self.full = bucket;

                Some(self)
            }
            Empty(..) => None
        }
    }
}


fn allocation_align<Z>() -> usize
    where Z: ZippedPtrs
{
    cmp::max(align_of::<u64>(), <Z as ZippedPtrs>::align())
}

fn allocation_size<Z: ZippedPtrs>(capacity: usize) -> usize {
    let hashes_size = capacity.checked_mul(size_of::<u64>())
                        .expect("capacity overflow");
    <Z as ZippedPtrs>::alloc_size_append(hashes_size, capacity)
        .expect("capacity overflow")
}

// Like allocation_size, but without any overflow checks
fn allocation_size_unchecked<Z: ZippedPtrs>(capacity: usize) -> usize {
    let hashes_size = capacity * size_of::<u64>();
    <Z as ZippedPtrs>::alloc_size_append_unchecked(hashes_size, capacity)
}

#[test]
fn test_offset_calculation() {

    use zipped::{LonePtr, PairPtrs};

    fn calculate_both_ways<Z: ZippedPtrs>(capacity: usize) -> usize {
        let size_checked = allocation_size::<Z>(capacity);
        let size_unchecked = allocation_size_unchecked::<Z>(capacity);
        assert_eq!(size_checked, size_unchecked);
        size_checked
    }

    assert_eq!(calculate_both_ways::<LonePtr<u16>>(7), 56 + 14);
    assert_eq!(calculate_both_ways::<PairPtrs<u8, u16>>(7), 56 + 8 + 14);
}

impl<Z: ZippedPtrs> RawTable<Z> {
    /// Does not initialize the buckets. The caller should ensure they,
    /// at the very least, set every hash to EMPTY_BUCKET.
    unsafe fn new_uninitialized(capacity: usize) -> RawTable<Z> {
        if capacity == 0 {
            return RawTable {
                size: 0,
                capacity: 0,
                hashes: Unique::new(EMPTY as *mut u64),
                marker: marker::PhantomData,
            };
        }

        // Allocating hash tables is a little tricky. We need to allocate
        // arrays for each column, but since we know their sizes and
        // alignments up front, we just allocate a single array, and then
        // have the subarrays point into it.
        //
        // This is great in theory, but in practice getting the alignment
        // right is a little subtle. Therefore, calculating offsets has been
        // factored out into a different function.
        let size = allocation_size::<Z>(capacity);
        let malloc_alignment = allocation_align::<Z>();

        let buffer = allocate(size, malloc_alignment);
        if buffer.is_null() { ::alloc::oom() }

        let hashes = buffer as *mut u64;

        RawTable {
            capacity: capacity,
            size:     0,
            hashes:   Unique::new(hashes),
            marker:   marker::PhantomData,
        }
    }

    fn first_bucket_raw(&self) -> RawBucket<Z> {
        let hashes_size = self.capacity * size_of::<u64>();
        let buffer = *self.hashes as *const u8;
        unsafe {
            let data = ZippedPtrs::from_unzipped_buf(buffer,
                                                     hashes_size,
                                                     self.capacity);
            RawBucket {
                hash: *self.hashes,
                data: data
            }
        }
    }

    /// Creates a new raw table from a given capacity. All buckets are
    /// initially empty.
    pub fn new(capacity: usize) -> RawTable<Z> {
        unsafe {
            let ret = RawTable::new_uninitialized(capacity);
            ptr::write_bytes(*ret.hashes, 0, capacity);
            ret
        }
    }

    /// The hashtable's capacity, similar to a vector's.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The number of elements ever `put` in the hashtable, minus the number
    /// of elements ever `take`n.
    pub fn size(&self) -> usize {
        self.size
    }

    fn raw_buckets(&self) -> RawBuckets<Z> {
        RawBuckets {
            raw: self.first_bucket_raw(),
            hashes_end: unsafe {
                self.hashes.offset(self.capacity as isize)
            },
            marker: marker::PhantomData,
        }
    }

    pub fn iter(&self) -> Iter<Z> {
        Iter {
            iter: self.raw_buckets(),
            elems_left: self.size(),
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<Z> {
        IterMut {
            iter: self.raw_buckets(),
            elems_left: self.size(),
            _marker: marker::PhantomData,
        }
    }

    pub fn into_iter(self) -> IntoIter<Z> {
        let RawBuckets { raw, hashes_end, .. } = self.raw_buckets();
        // Replace the marker regardless of lifetime bounds on parameters.
        IntoIter {
            iter: RawBuckets {
                raw: raw,
                hashes_end: hashes_end,
                marker: marker::PhantomData,
            },
            table: self,
        }
    }

    pub fn drain(&mut self) -> Drain<Z> {
        let RawBuckets { raw, hashes_end, .. } = self.raw_buckets();
        // Replace the marker regardless of lifetime bounds on parameters.
        Drain {
            iter: RawBuckets {
                raw: raw,
                hashes_end: hashes_end,
                marker: marker::PhantomData,
            },
            table: self,
        }
    }

    /// Returns an iterator that copies out each entry. Used while the table
    /// is being dropped.
    unsafe fn rev_move_buckets(&mut self) -> RevMoveBuckets<Z> {
        let raw_bucket = self.first_bucket_raw();
        RevMoveBuckets {
            raw: raw_bucket.offset(self.capacity as isize),
            hashes_end: raw_bucket.hash,
            elems_left: self.size,
            marker:     marker::PhantomData,
        }
    }
}

/// A raw iterator. The basis for some other iterators in this module. Although
/// this interface is safe, it's not used outside this module.
struct RawBuckets<'a, Z> {
    raw: RawBucket<Z>,
    hashes_end: *mut u64,

    // Strictly speaking, this should be `&'a Z`, but that would
    // require that Z:'a, and we often use RawBuckets<'static...> for
    // move iterations, so that messes up a lot of other things. So
    // just use `&'a ()` as this is not a publicly exposed type
    // anyway.
    marker: marker::PhantomData<&'a ()>,
}

impl<'a, Z: Copy> Clone for RawBuckets<'a, Z> {
    fn clone(&self) -> RawBuckets<'a, Z> {
        RawBuckets {
            raw: self.raw,
            hashes_end: self.hashes_end,
            marker: marker::PhantomData,
        }
    }
}


impl<'a, Z: ZippedPtrs> Iterator for RawBuckets<'a, Z> {
    type Item = RawBucket<Z>;

    fn next(&mut self) -> Option<RawBucket<Z>> {
        while self.raw.hash != self.hashes_end {
            unsafe {
                // We are swapping out the pointer to a bucket and replacing
                // it with the pointer to the next one.
                let prev = ptr::replace(&mut self.raw, self.raw.offset(1));
                if *prev.hash != EMPTY_BUCKET {
                    return Some(prev);
                }
            }
        }

        None
    }
}

/// An iterator that moves out buckets in reverse order. It leaves the table
/// in an inconsistent state and should only be used for dropping
/// the table's remaining entries. It's used in the implementation of Drop.
struct RevMoveBuckets<'a, Z> {
    raw: RawBucket<Z>,
    hashes_end: *mut u64,
    elems_left: usize,

    // As above, `&'a Z` would seem better, but we often use
    // 'static for the lifetime, and this is not a publicly exposed
    // type.
    marker: marker::PhantomData<&'a ()>,
}

impl<'a, Z: ZippedPtrs> Iterator for RevMoveBuckets<'a, Z> {
    type Item = Z::Values;

    fn next(&mut self) -> Option<Z::Values> {
        if self.elems_left == 0 {
            return None;
        }

        loop {
            debug_assert!(self.raw.hash != self.hashes_end);

            unsafe {
                self.raw = self.raw.offset(-1);

                if *self.raw.hash != EMPTY_BUCKET {
                    self.elems_left -= 1;
                    return Some(self.raw.data.read());
                }
            }
        }
    }
}

/// Iterator over shared references to entries in a table.
pub struct Iter<'a, Z> {
    iter: RawBuckets<'a, Z>,
    elems_left: usize,
}

unsafe impl<'a, Z> Sync for Iter<'a, Z>
    where Z: ZippedPtrs, Z::Values: Sync {}
unsafe impl<'a, Z> Send for Iter<'a, Z>
    where Z: ZippedPtrs, Z::Values: Sync {}

impl<'a, Z: Copy> Clone for Iter<'a, Z> {
    fn clone(&self) -> Iter<'a, Z> {
        Iter {
            iter: self.iter.clone(),
            elems_left: self.elems_left
        }
    }
}


/// Iterator over mutable references to entries in a table.
pub struct IterMut<'a, Z: ZippedPtrs> where Z::Values: 'a {
    iter: RawBuckets<'a, Z>,
    elems_left: usize,
    // To ensure invariance with respect to Z::Values
    _marker: marker::PhantomData<&'a mut Z::Values>,
}

unsafe impl<'a, Z> Sync for IterMut<'a, Z>
    where Z: ZippedPtrs, Z::Values: Sync {}
// Both Z::Values: Sync and Z::Values: Send are correct for IterMut's Send impl,
// but Send is the more useful bound
unsafe impl<'a, Z> Send for IterMut<'a, Z>
    where Z: ZippedPtrs, Z::Values: Send {}

/// Iterator over the entries in a table, consuming the table.
pub struct IntoIter<Z: ZippedPtrs> {
    table: RawTable<Z>,
    iter: RawBuckets<'static, Z>
}

unsafe impl<Z> Sync for IntoIter<Z> where Z: ZippedPtrs, Z::Values: Sync {}
unsafe impl<Z> Send for IntoIter<Z> where Z: ZippedPtrs, Z::Values: Send {}

/// Iterator over the entries in a table, clearing the table.
pub struct Drain<'a, Z: ZippedPtrs + 'a> {
    table: &'a mut RawTable<Z>,
    iter: RawBuckets<'static, Z>,
}

unsafe impl<'a, Z> Sync for Drain<'a, Z>
    where Z: ZippedPtrs, Z::Values: Sync {}
unsafe impl<'a, Z> Send for Drain<'a, Z>
    where Z: ZippedPtrs, Z::Values: Send {}

impl<'a, Z> Iterator for Iter<'a, Z> where Z: ZippedPtrs, Z::Values: 'a {
    type Item = Refs<'a, Z>;

    fn next(&mut self) -> Option<Refs<'a, Z>> {
        self.iter.next().map(|bucket| {
            self.elems_left -= 1;
            unsafe {
                Refs::from_ptrs(bucket.data)
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.elems_left, Some(self.elems_left))
    }
}
impl<'a, Z> ExactSizeIterator for Iter<'a, Z>
    where Z: ZippedPtrs, Z::Values: 'a
{
    fn len(&self) -> usize { self.elems_left }
}

impl<'a, Z> Iterator for IterMut<'a, Z> where Z: ZippedPtrs, Z::Values: 'a {
    type Item = MutRefs<'a, Z>;

    fn next(&mut self) -> Option<MutRefs<'a, Z>> {
        self.iter.next().map(|bucket| {
            self.elems_left -= 1;
            unsafe {
                MutRefs::from_ptrs(bucket.data)
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.elems_left, Some(self.elems_left))
    }
}
impl<'a, Z> ExactSizeIterator for IterMut<'a, Z>
    where Z: ZippedPtrs, Z::Values: 'a
{
    fn len(&self) -> usize { self.elems_left }
}

impl<Z: ZippedPtrs> Iterator for IntoIter<Z> {
    type Item = (SafeHash, Z::Values);

    fn next(&mut self) -> Option<(SafeHash, Z::Values)> {
        self.iter.next().map(|bucket| {
            self.table.size -= 1;
            unsafe {
                (
                    SafeHash {
                        hash: *bucket.hash,
                    },
                    bucket.data.read()
                )
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.table.size();
        (size, Some(size))
    }
}
impl<Z: ZippedPtrs> ExactSizeIterator for IntoIter<Z> {
    fn len(&self) -> usize { self.table.size() }
}

impl<'a, Z: ZippedPtrs> Iterator for Drain<'a, Z> {
    type Item = (SafeHash, Z::Values);

    #[inline]
    fn next(&mut self) -> Option<(SafeHash, Z::Values)> {
        self.iter.next().map(|bucket| {
            self.table.size -= 1;
            unsafe {
                (
                    SafeHash {
                        hash: ptr::replace(bucket.hash, EMPTY_BUCKET),
                    },
                    bucket.data.read()
                )
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.table.size();
        (size, Some(size))
    }
}
impl<'a, Z: ZippedPtrs> ExactSizeIterator for Drain<'a, Z> {
    fn len(&self) -> usize { self.table.size() }
}

impl<'a, Z: ZippedPtrs> Drop for Drain<'a, Z> {
    fn drop(&mut self) {
        for _ in self {}
    }
}

impl<Z> Clone for RawTable<Z>
    where Z: CloneZipped + Copy
{
    fn clone(&self) -> RawTable<Z> {
        unsafe {
            let mut new_ht: RawTable<Z> =
                RawTable::new_uninitialized(self.capacity());

            {
                let cap = self.capacity();
                let mut new_buckets = Bucket::first(&mut new_ht);
                let mut buckets = Bucket::first(self);
                while buckets.index() != cap {
                    match buckets.peek() {
                        Full(full) => {
                            let (h, v) = (full.hash(), full.clone_values());
                            *new_buckets.raw.hash = h.inspect();
                            new_buckets.raw.data.write(v);
                        }
                        Empty(..) => {
                            *new_buckets.raw.hash = EMPTY_BUCKET;
                        }
                    }
                    new_buckets.next();
                    buckets.next();
                }
            };

            new_ht.size = self.size();

            new_ht
        }
    }
}

impl<Z: ZippedPtrs> Drop for RawTable<Z> {
    //#[unsafe_destructor_blind_to_params]
    fn drop(&mut self) {
        if self.capacity == 0 || self.capacity == mem::POST_DROP_USIZE {
            return;
        }

        // This is done in reverse because we've likely partially taken
        // some elements out with `.into_iter()` from the front.
        // Check if the size is 0, so we don't do a useless scan when
        // dropping empty tables such as on resize.
        // Also avoid double drop of elements that have been already moved out.
        unsafe {
            if needs_drop::<Z::Values>() { // avoid linear runtime for types that don't need drop
                for _ in self.rev_move_buckets() {}
            }
        }

        // We have validated the size during initialization,
        // so unchecked calculation can be used now.
        let size = allocation_size_unchecked::<Z>(self.capacity);
        let align = allocation_align::<Z>();

        unsafe {
            deallocate(*self.hashes as *mut u8, size, align);
            // Remember how everything was allocated out of one buffer
            // during initialization? We only need one call to free here.
        }
    }
}
