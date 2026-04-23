---
layout: leetcode-entry
title: "1608. Special Array With X Elements Greater Than or Equal X"
permalink: "/leetcode/problem/2024-05-27-1608-special-array-with-x-elements-greater-than-or-equal-x/"
leetcode_ui: true
entry_slug: "2024-05-27-1608-special-array-with-x-elements-greater-than-or-equal-x"
---

[1608. Special Array With X Elements Greater Than or Equal X](https://leetcode.com/problems/special-array-with-x-elements-greater-than-or-equal-x/description/) easy
[blog post](https://leetcode.com/problems/special-array-with-x-elements-greater-than-or-equal-x/solutions/5213994/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27052024-1608-special-array-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/slW3XCHt4Ys)
![2024-05-27_07-27.webp](/assets/leetcode_daily_images/62c71c23.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/619

#### Problem TLDR

Count of more or equal nums[i] equal itself #easy

#### Intuition

Star with brute force, the `n` is in range `0..1000`, try them all, and for each count how many numbers are `nums[i] >= n`.

This will pass the checker.
Now time to optimize. If we sort the `nums` we can optimize the `nums[i] >= n`, as `n` only grows up so the `i`. We can start with the previous `i` next time.
Another optimizations, there are no more than `nums.size` count possible, so `n`'s range is `0..nums.size` inclusive.

#### Approach

Let's write non-optimal one-liner in Kotlin, and more robust solution in Rust.

#### Complexity

- Time complexity:
$$O(nlogn)$$ and $$O(n^2)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun specialArray(nums: IntArray): Int = (0..nums.size)
        .firstOrNull { n -> n == nums.count { it >= n }} ?: -1

```
```rust

    pub fn special_array(mut nums: Vec<i32>) -> i32 {
        nums.sort_unstable(); let (mut n, mut i) = (0, 0);
        for n in 0..=nums.len() {
            while i < nums.len() && nums[i] < n as i32 { i += 1 }
            if n == nums.len() - i { return n as i32 }
        }; -1
    }

```

