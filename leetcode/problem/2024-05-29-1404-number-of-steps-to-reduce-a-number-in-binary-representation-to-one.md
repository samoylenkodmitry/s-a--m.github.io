---
layout: leetcode-entry
title: "1404. Number of Steps to Reduce a Number in Binary Representation to One"
permalink: "/leetcode/problem/2024-05-29-1404-number-of-steps-to-reduce-a-number-in-binary-representation-to-one/"
leetcode_ui: true
entry_slug: "2024-05-29-1404-number-of-steps-to-reduce-a-number-in-binary-representation-to-one"
---

[1404. Number of Steps to Reduce a Number in Binary Representation to One](https://leetcode.com/problems/number-of-steps-to-reduce-a-number-in-binary-representation-to-one/description/) medium
[blog post](https://leetcode.com/problems/number-of-steps-to-reduce-a-number-in-binary-representation-to-one/solutions/5224598/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29052024-1404-number-of-steps-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/kGeMgXTgP8M)
![2024-05-29_09-04.webp](/assets/leetcode_daily_images/c03768cb.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/621

#### Problem TLDR

Steps `even/2`, `odd+1` to make binary `s` to `1` #medium

#### Intuition

We can just implement what is asked recursively passing a new string each time.
The more interesting and effective solution is to iterate from the end and try to count operations on the fly:
* calculate `current` and `carry`
* apply extra operation if `current` is `odd` and do extra increase for carry

#### Approach

Let's minify the code using the math tricks.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun numSteps(s: String): Int {
        var carry = 0
        return (s.lastIndex downTo 1).sumOf { i ->
            val curr = s[i] - '0' + carry
            carry = curr / 2 + curr % 2
            1 + curr % 2
        } + carry
    }

```
```rust

    pub fn num_steps(s: String) -> i32 {
        let (mut carry, sb) = (0, s.as_bytes());
        (1..s.len()).rev().map(|i| {
            let curr = sb[i] as i32 - b'0' as i32 + carry;
            carry = curr / 2 + curr % 2;
            1 + curr % 2
        }).sum::<i32>() + carry
    }

```

