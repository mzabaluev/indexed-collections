// Copyright © 2016  Mikhail Zabaluev <mikhail.zabaluev@gmail.com>
// See the COPYRIGHT file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(alloc)]
#![feature(core_intrinsics)]
#![feature(dropck_parametricity)]
#![feature(filling_drop)]
#![feature(heap_api)]
#![feature(oom)]
#![feature(unique)]
#![feature(unsafe_no_drop_flag)]

extern crate alloc;
extern crate rand;

mod hash;

pub mod hash_map {
    pub use super::hash::map::*;
}

mod zipped;
