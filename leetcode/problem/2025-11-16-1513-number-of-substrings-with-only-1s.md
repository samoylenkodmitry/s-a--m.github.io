---
layout: leetcode-entry
title: "1513. Number of Substrings With Only 1s"
permalink: "/leetcode/problem/2025-11-16-1513-number-of-substrings-with-only-1s/"
leetcode_ui: true
entry_slug: "2025-11-16-1513-number-of-substrings-with-only-1s"
---

[1513. Number of Substrings With Only 1s](https://leetcode.com/problems/number-of-substrings-with-only-1s/description/) medium
[blog post](https://leetcode.com/problems/number-of-substrings-with-only-1s/solutions/7352028/kotlin-rust-by-samoylenkodmitry-2m4m/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16112025-1513-number-of-substrings?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/egWge3s-qBU)

![5e173ed0-6bd3-4f9b-99fa-250efe0725c7 (1).webp](/assets/leetcode_daily_images/96d2c063.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1175

#### Problem TLDR

Substrings of ones #medium #counting

#### Intuition

Sum the separate islands of ones. Each island length is n(n+1)/2 arithmetic sum.

#### Approach

* or compute the arithmetic sum on the go

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 21ms
    fun numSub(s: String) = s.fold(0 to 0) { (r,curr), c ->
        val curr = (c-'0')*curr + (c-'0')
        (r + curr) % 1000000007 to curr
    }.first
```
```rust
// 0ms
    pub fn num_sub(s: String) -> i32 {
        s.bytes().fold((0, 0), |(r, curr), c| {
            let curr = (c-b'0')as i32*curr + (c-b'0')as i32;
            ((r+curr)%1000000007, curr)
        }).0
    }
```

