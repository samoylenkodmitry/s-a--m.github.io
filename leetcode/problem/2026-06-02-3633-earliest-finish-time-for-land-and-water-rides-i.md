---
layout: leetcode-entry
title: "3633. Earliest Finish Time for Land and Water Rides I"
permalink: "/leetcode/problem/2026-06-02-3633-earliest-finish-time-for-land-and-water-rides-i/"
leetcode_ui: true
entry_slug: "2026-06-02-3633-earliest-finish-time-for-land-and-water-rides-i"
---

[3633. Earliest Finish Time for Land and Water Rides I](https://leetcode.com/problems/earliest-finish-time-for-land-and-water-rides-i/solutions/8307823/kotlin-rust-by-samoylenkodmitry-usfy/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02062026-3633-earliest-finish-time?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/QOgbGrliyAw)

https://dmitrysamoylenko.com/leetcode/

![02.06.2026.webp](/assets/leetcode_daily_images/02.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1378

#### Problem TLDR

Min finish time of land + water single events

#### Intuition

Brute-force: take every water even and compare with every land event
Optimal: find min water and land finish times, then try all waters with finish land and try all lands with finish water

#### Approach

* extract the repeating parts

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun earliestFinishTime(lst: IntArray, ld: IntArray, wst: IntArray, wd: IntArray): Int {
        val f = { s:IntArray,d:IntArray,m:Int -> s.indices.minOf{max(m,s[it])+d[it]}}
        return min(f(wst,wd, f(lst,ld,0)),f(lst,ld,f(wst,wd,0)))
    }
```
```rust
    pub fn earliest_finish_time(lst: Vec<i32>, ld: Vec<i32>, wst: Vec<i32>, wd: Vec<i32>) -> i32 {
        let f = |s: &[i32], d: &[i32], m: i32| (0..s.len()).map(|i| s[i].max(m) + d[i]).min().unwrap();
        f(&wst, &wd, f(&lst, &ld, 0)).min(f(&lst, &ld, f(&wst, &wd, 0)))
    }
```

