---
layout: leetcode-entry
title: "1052. Grumpy Bookstore Owner"
permalink: "/leetcode/problem/2024-06-21-1052-grumpy-bookstore-owner/"
leetcode_ui: true
entry_slug: "2024-06-21-1052-grumpy-bookstore-owner"
---

[1052. Grumpy Bookstore Owner](https://leetcode.com/problems/grumpy-bookstore-owner/description/) medium
[blog post](https://leetcode.com/problems/grumpy-bookstore-owner/solutions/5344521/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21062024-1052-grumpy-bookstore-owner?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/VOgbvWK3myc)
![2024-06-21_07-07_1.webp](/assets/leetcode_daily_images/8f7ff95a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/646

#### Problem TLDR

Max customers sum after make consecutive `minutes` non-grumpy #medium #sliding_window

#### Intuition

It was hard.
First understand the problem: we can take all the `0`-grumpy minutes, but `1`-grumpy can only be in `minutes`, and must be choosen.
Let's explore the example:

```j

    // 1  2  3 4  5  6 7  8 9      m=2
    // 1  1  0 1  1  0 1  1 1
    // *  *    *  *    *  *
    //                    * *
    //
    // 2 4 1 4 1   m=2
    // 1 0 1 0 1
    // * *
    //     * *

```
The `sliding window` must be from the `1-grumpy` days to choose the maximum and ignore all `0-grumpy` days, because they are always be taken.

#### Approach

Keep `0`-grumpy and `1` grumpy sums in separate variables.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun maxSatisfied(customers: IntArray, grumpy: IntArray, minutes: Int): Int {
        var sum = 0; var max = 0; var other = 0; var j = 0
        for ((i, c) in customers.withIndex()) {
            sum += c * grumpy[i]
            other += c * (1 - grumpy[i])
            while (j <= i - minutes) sum -= customers[j] * grumpy[j++]
            max = max(max, sum)
        }
        return max + other
    }

```
```rust

    pub fn max_satisfied(customers: Vec<i32>, grumpy: Vec<i32>, minutes: i32) -> i32 {
        let (mut j, mut sum, mut other, mut max) = (0, 0, 0, 0);
        for i in 0..grumpy.len() {
            other += customers[i] * (1 - grumpy[i]);
            sum += customers[i] * grumpy[i];
            while j as i32 <= i as i32 - minutes { sum -= customers[j] * grumpy[j]; j += 1 }
            max = max.max(sum)
        }; max + other
    }

```

