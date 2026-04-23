---
layout: leetcode-entry
title: "2011. Final Value of Variable After Performing Operations"
permalink: "/leetcode/problem/2025-10-20-2011-final-value-of-variable-after-performing-operations/"
leetcode_ui: true
entry_slug: "2025-10-20-2011-final-value-of-variable-after-performing-operations"
---

[2011. Final Value of Variable After Performing Operations](https://leetcode.com/problems/final-value-of-variable-after-performing-operations/description) easy
[blog post](https://leetcode.com/problems/final-value-of-variable-after-performing-operations/solutions/7287908/kotlin-rust-by-samoylenkodmitry-m3j5/)
[substack]()
[youtube](https://youtu.be/FBnBfpq2wnc)

![1476e184-c73b-4b5f-a496-0feaf1535283 (1).webp](/assets/leetcode_daily_images/27105282.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1148

#### Problem TLDR

do ++ or -- operation from 0 #easy

#### Intuition

Simulate the process.

#### Approach

* just check '+' in string

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 7ms
    fun finalValueAfterOperations(o: Array<String>) =
        2 * o.count { '+' in it } - o.size

```
```rust

// 1ms
    pub fn final_value_after_operations(o: Vec<String>) -> i32 {
        (o.join("").matches('+').count() - o.len()) as _
    }

```

