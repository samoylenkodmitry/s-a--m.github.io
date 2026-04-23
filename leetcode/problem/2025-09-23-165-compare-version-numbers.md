---
layout: leetcode-entry
title: "165. Compare Version Numbers"
permalink: "/leetcode/problem/2025-09-23-165-compare-version-numbers/"
leetcode_ui: true
entry_slug: "2025-09-23-165-compare-version-numbers"
---

[165. Compare Version Numbers](https://leetcode.com/problems/compare-version-numbers/description) medium
[blog post](https://leetcode.com/problems/compare-version-numbers/solutions/7216721/kotlin-rust-by-samoylenkodmitry-dtgc/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23092025-165-compare-version-numbers?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/-qstZnKVbNs)

![1.webp](/assets/leetcode_daily_images/8c29e723.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1121

#### Problem TLDR

Compare versions x.x.x.x #medium

#### Intuition

Pad start strings or convert to ints.

#### Approach

* 25 characters for pad start
* pad lists of numbers length to the largest

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 23ms
    fun compareVersion(v1: String, v2: String) = listOf(v1, v2)
        .map { it.split('.').map {it.toInt()}}
        .let { (a, b) -> val d = List(abs(a.size - b.size)){0}; (a+d).zip(b+d)}
        .map { (a, b) -> a.compareTo(b) }.firstOrNull { it != 0 } ?: 0

```

```rust

// 0ms
    pub fn compare_version(v: String, w: String) -> i32 {
        v.split('.').zip_longest(w.split('.')).map(|e|e.or("0","0"))
        .map(|(l,r)|l.parse::<i32>().unwrap().cmp(&r.parse::<i32>().unwrap()) as i32)
        .find(|&x| x != 0).unwrap_or(0)
    }

```

