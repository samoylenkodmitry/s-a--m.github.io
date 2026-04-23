---
layout: leetcode-entry
title: "624. Maximum Distance in Arrays"
permalink: "/leetcode/problem/2024-08-16-624-maximum-distance-in-arrays/"
leetcode_ui: true
entry_slug: "2024-08-16-624-maximum-distance-in-arrays"
---

[624. Maximum Distance in Arrays](https://leetcode.com/problems/maximum-distance-in-arrays/description/) medium
[blog post](https://leetcode.com/problems/maximum-distance-in-arrays/solutions/5643627/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16082024-624-maximum-distance-in?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/F4VCW4tqreM)
![1.webp](/assets/leetcode_daily_images/9de66bce.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/704

#### Problem TLDR

Max diff between the arrays #medium

#### Intuition

We must not use the `min` and `max` from the same array, that is the main problem here.

The ugly way to do this is to find the `min` and `second min` and same for `max`, then compare it with the current array in the second pass.

There is a one pass solution, however, and it looks much nicer. Just not use the current `min` and `max` simultaneously.

#### Approach

We can save some lines of code with iterators.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    var min = Int.MAX_VALUE / 2; var max = -min
    fun maxDistance(arrays: List<List<Int>>) = arrays
        .maxOf { a ->
            maxOf(max - a[0], a.last() - min)
            .also { max = max(max, a.last()); min = min(min, a[0]) }
        }

```
```rust

    pub fn max_distance(arrays: Vec<Vec<i32>>) -> i32 {
        let (mut min, mut max) = (i32::MAX / 2, i32::MIN / 2);
        arrays.iter().map(|a| {
            let diff = (max - a[0]).max(a[a.len() - 1] - min);
            max = max.max(a[a.len() - 1]); min = min.min(a[0]); diff
        }).max().unwrap()
    }

```

