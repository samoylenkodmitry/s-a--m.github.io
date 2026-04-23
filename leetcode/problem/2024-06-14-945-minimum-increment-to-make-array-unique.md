---
layout: leetcode-entry
title: "945. Minimum Increment to Make Array Unique"
permalink: "/leetcode/problem/2024-06-14-945-minimum-increment-to-make-array-unique/"
leetcode_ui: true
entry_slug: "2024-06-14-945-minimum-increment-to-make-array-unique"
---

[945. Minimum Increment to Make Array Unique](https://leetcode.com/problems/minimum-increment-to-make-array-unique/description/) medium
[blog post](https://leetcode.com/problems/minimum-increment-to-make-array-unique/solutions/5310347/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14062024-945-minimum-increment-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/3hosp15Hy_8)
![2024-06-14_06-25_1.webp](/assets/leetcode_daily_images/6f693415.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/639

#### Problem TLDR

Min increments to make items unique #medium

#### Intuition

Let's observe an example.
```j
    // 1 2 2         delta   diff
    //   i           0       1
    //     i         1       0
    //
    // 1 1 2 2 3 7
    //   i           1       0
    //     i         1       1
    //       i       2       0
    //         i     2       1
    //           i   0       4
    //              (2 - (4-1))
```
First, sort, then maintain the `delta`:
* increase if there is a duplicate
* decrease by adjucent items diff

#### Approach

Let's use iterators: `windowed`, `sumOf`.

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$, but O(n) for `sorted` in Kotlin

#### Code

```kotlin

    fun minIncrementForUnique(nums: IntArray): Int {
        var delta = 0
        return nums.sorted().windowed(2).sumOf { (a, b) ->
            if (a < b) delta = max(0, delta + a - b + 1) else delta++
            delta
        }
    }

```
```rust

    pub fn min_increment_for_unique(mut nums: Vec<i32>) -> i32 {
        nums.sort_unstable(); let mut delta = 0;
        nums[..].windows(2).map(|w| {
            delta = if w[0] < w[1] { 0.max(delta + w[0] - w[1] + 1) } else { delta + 1 };
            delta
        }).sum()
    }

```

