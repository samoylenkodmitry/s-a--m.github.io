---
layout: leetcode-entry
title: "2048. Next Greater Numerically Balanced Number"
permalink: "/leetcode/problem/2025-10-24-2048-next-greater-numerically-balanced-number/"
leetcode_ui: true
entry_slug: "2025-10-24-2048-next-greater-numerically-balanced-number"
---

[2048. Next Greater Numerically Balanced Number](https://leetcode.com/problems/next-greater-numerically-balanced-number/description/?) medium
[blog post](https://leetcode.com/problems/next-greater-numerically-balanced-number/solutions/7296867/kotlin-rust-by-samoylenkodmitry-keh9/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24102025-2048-next-greater-numerically?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ym9PeBjBVRc)

![18ff3401-d0bb-4fd6-9898-57577a733eec (1).webp](/assets/leetcode_daily_images/18ca2032.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1152

#### Problem TLDR

First bigger number freq[digit]=digit #medium

#### Intuition

Brute-force is accepted.
For the problem size of 10^6 the next value is 1224444 which is just 200k loop.

#### Approach

* or we can try to generate all permutations; prune by length of the initial number

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 425ms
    fun nextBeautifulNumber(n: Int) = (n+1..n*21+1)
        .first { "$it".groupBy { it }.all { it.key-'0' == it.value.size }}

```
```rust

// 22ms
    pub fn next_beautiful_number(n: i32) -> i32 {
        (n+1..n*22+2).find(|&x| { let (mut y,mut f) = (x, [0;10]);
            while y > 0 { f[(y%10)as usize] += 1; y /= 10 }
            (0..=9).all(|x| f[x] == x || f[x] < 1)
        }).unwrap()
    }

```

