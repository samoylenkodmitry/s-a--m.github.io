---
layout: leetcode-entry
title: "3635. Earliest Finish Time for Land and Water Rides II"
permalink: "/leetcode/problem/2026-06-03-3635-earliest-finish-time-for-land-and-water-rides-ii/"
leetcode_ui: true
entry_slug: "2026-06-03-3635-earliest-finish-time-for-land-and-water-rides-ii"
---

[3635. Earliest Finish Time for Land and Water Rides II](https://leetcode.com/problems/earliest-finish-time-for-land-and-water-rides-ii/solutions/8310458/kotlin-rust-by-samoylenkodmitry-88y9/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03062026-3635-earliest-finish-time?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ctbEphoD83c)

https://dmitrysamoylenko.com/leetcode/

![03.06.2026.webp](/assets/leetcode_daily_images/03.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1379

#### Problem TLDR

Min finish time of land + water single events

#### Intuition

find min water and land finish times, then try all waters with finish land and try all lands with finish water

#### Approach

* extract the repeating parts
* Rust can derive the argument types in lambdas
* Kotlin: we can extract sub-sub functions

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun earliestFinishTime(lst: IntArray, ld: IntArray, wst: IntArray, wd: IntArray)= run {
        fun f(st: IntArray, d: IntArray) = {s: Int ->st.zip(d).minOf{(st,d)->max(s,st)+d}}
        val (a,b) = f(lst, ld) to f(wst, wd); min(a(b(0)), b(a(0)))
    }
```
```rust
    pub fn earliest_finish_time(lst: Vec<i32>, ld: Vec<i32>, wst: Vec<i32>, wd: Vec<i32>) -> i32 {
        let f = |st:&[i32],d:&[i32], s| (0..d.len()).map(|i|st[i].max(s)+d[i]).min().unwrap();
        f(&lst, &ld, f(&wst, &wd, 0)).min(f(&wst, &wd, f(&lst, &ld, 0)))
    }
```

