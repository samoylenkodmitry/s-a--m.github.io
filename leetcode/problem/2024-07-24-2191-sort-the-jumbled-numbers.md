---
layout: leetcode-entry
title: "2191. Sort the Jumbled Numbers"
permalink: "/leetcode/problem/2024-07-24-2191-sort-the-jumbled-numbers/"
leetcode_ui: true
entry_slug: "2024-07-24-2191-sort-the-jumbled-numbers"
---

[2191. Sort the Jumbled Numbers](https://leetcode.com/problems/sort-the-jumbled-numbers/description/) medium
[blog post](https://leetcode.com/problems/sort-the-jumbled-numbers/solutions/5526254/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24072024-2191-sort-the-jumbled-numbers?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/xW8ZM1CQpg8)
![2024-07-24_08-29_1.webp](/assets/leetcode_daily_images/8013c887.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/680

#### Problem TLDR

Sort array by digits mapping #medium

#### Intuition

Just sort using a comparator by key

#### Approach

* careful with the corner case n = 0
* in Rust using `sort_by_cached_key` has improved runtime from 170ms to 20ms

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$, O(n) for Kotlin, as it didn't have a proper sorting method for IntArray

#### Code

```kotlin

    fun sortJumbled(mapping: IntArray, nums: IntArray) =
        nums.sortedWith(compareBy {
            var n = it
            var res = if (n < 1) mapping[n] else 0
            var pow = 1
            while (n > 0) {
                res += pow * mapping[n % 10]
                pow *= 10
                n /= 10
            }
            res
        })

```
```rust

    pub fn sort_jumbled(mapping: Vec<i32>, mut nums: Vec<i32>) -> Vec<i32> {
        nums.sort_by_cached_key(|&x| {
            let (mut n, mut pow, mut res) = (x as usize, 1, 0);
            if x < 1 { res = mapping[n] }
            while n > 0 {
                res += pow * mapping[n % 10];
                pow *= 10; n /= 10
            }
            res
        }); nums
    }

```

