---
layout: leetcode-entry
title: "3075. Maximize Happiness of Selected Children"
permalink: "/leetcode/problem/2025-12-25-3075-maximize-happiness-of-selected-children/"
leetcode_ui: true
entry_slug: "2025-12-25-3075-maximize-happiness-of-selected-children"
---

[3075. Maximize Happiness of Selected Children](https://leetcode.com/problems/maximize-happiness-of-selected-children/description) medium
[blog post](https://leetcode.com/problems/maximize-happiness-of-selected-children/solutions/7437881/kotlin-rust-by-samoylenkodmitry-dh7g/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25122025-3075-maximize-happiness?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/pVXiXEODprI)

![75aecedd-9c48-4d4c-8be0-6480c83821ea (1).webp](/assets/leetcode_daily_images/6c4af9af.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1215

#### Problem TLDR

K max values, decreas each pick #medium #quickselect

#### Intuition

Sort and pick K largest.

#### Approach

* optimize with quickselect
* we still have to sort K items to check 0 overflow

#### Complexity

- Time complexity:
$$O(n + klog(k))$$, for optimal; the simplest is O(nlogn)

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 693ms
    fun maximumHappinessSum(h: IntArray, k: Int) =
        h.sortedDescending().take(k).withIndex().sumOf {(i,h)->max(0,1L*h-i)}
```
```rust
// 12ms
    pub fn maximum_happiness_sum(mut h: Vec<i32>, k: i32) -> i64 {
        h.sort_unstable_by(|a,b|b.cmp(a));
        h.iter().zip(0..k).map(|(h,i)|0.max(h-i)as i64).sum()
    }
```
```rust
// 18ms
    pub fn maximum_happiness_sum(mut h: Vec<i32>, k: i32) -> i64 {
        let (l,m,_) = h.select_nth_unstable_by_key(k as usize-1, |h|-h);
        l.sort_unstable_by(|a,b|b.cmp(a));
        l.iter().chain(once(&*m)).zip(0..).map(|(h,i)|0.max(h-i)as i64).sum()
    }
```

