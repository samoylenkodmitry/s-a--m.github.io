---
layout: leetcode-entry
title: "1846. Maximum Element After Decreasing and Rearranging"
permalink: "/leetcode/problem/2026-06-28-1846-maximum-element-after-decreasing-and-rearranging/"
leetcode_ui: true
entry_slug: "2026-06-28-1846-maximum-element-after-decreasing-and-rearranging"
---

[1846. Maximum Element After Decreasing and Rearranging](https://leetcode.com/problems/maximum-element-after-decreasing-and-rearranging/solutions/8363244/kotlin-rust-by-samoylenkodmitry-zlzj/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28062026-1846-maximum-element-after?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/EeSZMRdxiM0)

https://dmitrysamoylenko.com/leetcode/

![28.06.2026.webp](/assets/leetcode_daily_images/28.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1404

#### Problem TLDR

Max possible grow of +0 or +1

#### Intuition

Sort. Iterate. Take at most prev + 1.

#### Approach

* can it be solved in O(n)? yes, count sort, the max value is the arr.size

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun maximumElementAfterDecrementingAndRearranging(a: IntArray) =
    a.sorted().fold(0) { r, t -> min(r+1,t) }
```
```rust
    pub fn maximum_element_after_decrementing_and_rearranging(a: Vec<i32>) -> i32 {
        a.into_iter().sorted().fold(0,|r,t|t.min(r+1))
    }
```

