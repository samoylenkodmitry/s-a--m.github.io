---
layout: leetcode-entry
title: "2169. Count Operations to Obtain Zero"
permalink: "/leetcode/problem/2025-11-09-2169-count-operations-to-obtain-zero/"
leetcode_ui: true
entry_slug: "2025-11-09-2169-count-operations-to-obtain-zero"
---

[2169. Count Operations to Obtain Zero](https://leetcode.com/problems/count-operations-to-obtain-zero/description/) easy
[blog post](https://leetcode.com/problems/count-operations-to-obtain-zero/solutions/7336766/kotlin-rust-by-samoylenkodmitry-e5iz/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09112025-2169-count-operations-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/EQUm48CGOs0)

![ed52662d-317d-4c76-9e3f-e3fcd2ee7132 (1).webp](/assets/leetcode_daily_images/6e1c030b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1168

#### Problem TLDR

Count numbers subtractions until 0 #easy

#### Intuition

Simulate the process.

Another way to look at it: the biggest would be subtracted `big/small` times until numbers flip.

#### Approach

* recursion space complexity is log(depth)
* it is Euclidian algorithm (a,b) to (b, a%b)
* there is a golden ratio hidden here
* consequtive Fibonacci numbers would give the slowest path: Fibonacci sequence backwards

#### Complexity

- Time complexity:
$$O(log(n))$$

- Space complexity:
$$O(log(n))$$

#### Code

```kotlin
// 0ms
    fun countOperations(a: Int, b: Int): Int =
        if (a<1||b<1) 0 else a/b + countOperations(b,a%b)
```
```rust
// 0ms
    pub fn count_operations(a: i32, b: i32) -> i32 {
        if a<1 || b<1 { 0 } else { a/b + Self::count_operations(b,a%b) }
    }
```

