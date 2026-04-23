---
layout: leetcode-entry
title: "1404. Number of Steps to Reduce a Number in Binary Representation to One"
permalink: "/leetcode/problem/2026-02-26-1404-number-of-steps-to-reduce-a-number-in-binary-representation-to-one/"
leetcode_ui: true
entry_slug: "2026-02-26-1404-number-of-steps-to-reduce-a-number-in-binary-representation-to-one"
---

[1404. Number of Steps to Reduce a Number in Binary Representation to One](https://open.substack.com/pub/dmitriisamoilenko/p/26022026-1404-number-of-steps-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) medium
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/26022026-1404-number-of-steps-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26022026-1404-number-of-steps-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/BHQNA5xMXKk)

![22a9c953-789e-4786-8de8-7160633fdc92 (1).webp](/assets/leetcode_daily_images/83352c31.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1281

#### Problem TLDR

Ops /2+1 to make 1 #medium #simulation

#### Intuition

Simulate the process, O(n^2) is accepted

#### Approach

* to optimize propagate the carry to the next op instead of doing full +1 operation

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 10ms
    fun numSteps(s: String) = s.lastIndexOf('1') +
    (if (Regex("^10*$") in s) 0 else 2) + s.count {it<'1'}
```
```rust
// 0ms
    pub fn num_steps(s: String) -> i32 {
        let mut c = 0;
        (1..s.len()).rev().map(|i| {
            c += (s.as_bytes()[i] - b'0') as i32;
            let r = 1 + c % 2; c = c/2+c%2; r
        }).sum::<i32>() + c
    }
```

