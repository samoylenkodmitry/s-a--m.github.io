---
layout: leetcode-entry
title: "1208. Get Equal Substrings Within Budget"
permalink: "/leetcode/problem/2024-05-28-1208-get-equal-substrings-within-budget/"
leetcode_ui: true
entry_slug: "2024-05-28-1208-get-equal-substrings-within-budget"
---

[1208. Get Equal Substrings Within Budget](https://leetcode.com/problems/get-equal-substrings-within-budget/description/) medium
[blog post](https://leetcode.com/problems/get-equal-substrings-within-budget/solutions/5219126/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28052024-1208-get-equal-substrings?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Qy0xh319YHA)
![2024-05-28_07-23.webp](/assets/leetcode_daily_images/ab7467c7.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/620

#### Problem TLDR

Max substring `sum(abs(s[..] - t[..])) < maxCost` #medium #sliding_window

#### Intuition

There is a known `Sliding Window` technique to find any `max` or `min` in a *sub*string or *sub*array (contiguous part): use one pointer to take one more element on the right border, compute the result, then if there are some conditions, move the left border and recompute the result again. This will find the maximum while not checking *every* possible subarray: because we check all subarrays *ends* borders and we drop every *start* border that are clearly out of scope by `max` function.

#### Approach

* maxOf in Kotlin and .map().max() in Rust will help to save some lines of code

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun equalSubstring(s: String, t: String, maxCost: Int): Int {
        var i = 0; var cost = 0
        return s.indices.maxOf {
            cost += abs(s[it] - t[it])
            if (cost > maxCost) cost -= abs(s[i] - t[i++])
            it - i + 1
        }
    }

```
```rust

    pub fn equal_substring(s: String, t: String, max_cost: i32) -> i32 {
        let (mut i, mut cost, sb, tb) = (0, 0, s.as_bytes(), t.as_bytes());
        (0..s.len()).map(|j| {
            cost += (sb[j] as i32 - tb[j] as i32).abs();
            if cost > max_cost { cost -= (sb[i] as i32 - tb[i] as i32).abs(); i += 1 }
            j - i + 1
        }).max().unwrap() as _
    }

```

