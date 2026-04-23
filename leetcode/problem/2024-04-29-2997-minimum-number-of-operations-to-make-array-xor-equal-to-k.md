---
layout: leetcode-entry
title: "2997. Minimum Number of Operations to Make Array XOR Equal to K"
permalink: "/leetcode/problem/2024-04-29-2997-minimum-number-of-operations-to-make-array-xor-equal-to-k/"
leetcode_ui: true
entry_slug: "2024-04-29-2997-minimum-number-of-operations-to-make-array-xor-equal-to-k"
---

[2997. Minimum Number of Operations to Make Array XOR Equal to K](https://leetcode.com/problems/minimum-number-of-operations-to-make-array-xor-equal-to-k/description/) medium
[blog post](https://leetcode.com/problems/minimum-number-of-operations-to-make-array-xor-equal-to-k/solutions/5086260/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29042024-2997-minimum-number-of-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/kTTy66sDiIU)
![2024-04-29_07-41.webp](/assets/leetcode_daily_images/9b99ae4f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/587

#### Problem TLDR

Bit diff between `k` and `nums` xor #medium #bit_manipulation

#### Intuition

Let's observe how the result `xor` built:
```j
    // 2  010 -> 110
    // 1  001
    // 3  011 -> 010
    // 4  100
    // x  100 -> 000 -> 001
    // k  001
```
The result `x` differs from `k` by two bit flips: `100 -> 000 -> 001`. We can do those bit flips on any number in the array, the final `xor` does not depend on the number choice.

#### Approach

Let's try to use built-in methods: `fold`, `countOneBits`, `count_ones`.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minOperations(nums: IntArray, k: Int) =
        nums.fold(k) { r, t -> r xor t }.countOneBits()

```
```rust

    pub fn min_operations(nums: Vec<i32>, k: i32) -> i32 {
        nums.iter().fold(k, |r, t| r ^ t).count_ones() as _
    }

```

