---
layout: leetcode-entry
title: "2540. Minimum Common Value"
permalink: "/leetcode/problem/2026-05-19-2540-minimum-common-value/"
leetcode_ui: true
entry_slug: "2026-05-19-2540-minimum-common-value"
---

[2540. Minimum Common Value](https://leetcode.com/problems/minimum-common-value/solutions/8275223/kotlin-rust-by-samoylenkodmitry-b5vo/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19052026-2540-minimum-common-value?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/KTb5zzlSwQ0)

https://dmitrysamoylenko.com/leetcode/

![19.05.2026.webp](/assets/leetcode_daily_images/19.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1364

#### Problem TLDR

Min common of two sorted list

#### Intuition

* two pointers: one goes forward, second tries to match
* or hashset
* or binary search

#### Approach

* binary search is the shortest
* rust itertools has nice way to merge sorted lists

#### Complexity

- Time complexity:
$$O(nlogn|n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun getCommon(a:IntArray, b: IntArray) =
    a.find { b.binarySearch(it) >= 0 } ?: -1
```
```rust
    pub fn get_common(a: Vec<i32>, b: Vec<i32>) -> i32 {
        a.into_iter().merge_join_by(b, i32::cmp).find_map(|e|
            match e {Both(a,_)=>Some(a), _=>None}).unwrap_or(-1)
    }
```

