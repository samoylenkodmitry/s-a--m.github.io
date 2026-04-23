---
layout: leetcode-entry
title: "1380. Lucky Numbers in a Matrix"
permalink: "/leetcode/problem/2024-07-19-1380-lucky-numbers-in-a-matrix/"
leetcode_ui: true
entry_slug: "2024-07-19-1380-lucky-numbers-in-a-matrix"
---

[1380. Lucky Numbers in a Matrix](https://leetcode.com/problems/lucky-numbers-in-a-matrix/description/) easy
[blog post](https://leetcode.com/problems/lucky-numbers-in-a-matrix/solutions/5499164/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19072024-1380-lucky-numbers-in-a?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/YeTq-68Fo30)
![2024-07-19_08-22_1.webp](/assets/leetcode_daily_images/1f1510c4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/675

#### Problem TLDR

Min in rows and max in columns in a unique number matrix #easy

#### Intuition

As all the numbers are unique, we can first find all the maximums in the columns, then intersect the result with all the minimums in the rows.

#### Approach

Let's use the collections API's:
* maxOf, map, filter

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

    fun luckyNumbers (matrix: Array<IntArray>) = (0..<matrix[0].size)
        .map { x -> matrix.maxOf { it[x] }}.toSet().let { maxes ->
            matrix.map { it.min() }.filter { it in maxes }}

```
```rust

    pub fn lucky_numbers (matrix: Vec<Vec<i32>>) -> Vec<i32> {
        let maxes: Vec<_> = (0..matrix[0].len())
            .map(|x| matrix.iter().map(|r| r[x]).max().unwrap()).collect();
        matrix.iter().map(|r| *r.iter().min().unwrap())
            .filter(|v| maxes.contains(v)).collect()
    }

```

