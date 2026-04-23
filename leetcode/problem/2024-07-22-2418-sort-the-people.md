---
layout: leetcode-entry
title: "2418. Sort the People"
permalink: "/leetcode/problem/2024-07-22-2418-sort-the-people/"
leetcode_ui: true
entry_slug: "2024-07-22-2418-sort-the-people"
---

[2418. Sort the People](https://leetcode.com/problems/sort-the-people/description/) easy
[blog post](https://leetcode.com/problems/sort-the-people/solutions/5514920/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22072024-2418-sort-the-people?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/pw5pyrufU-8)
![2024-07-22_08-22_1.webp](/assets/leetcode_daily_images/f07b367e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/678

#### Problem TLDR

Sort one array by another #easy

#### Intuition

We must use some extra memory for the relations between the arrays: it can be an indices array, or a zipped collection. Then sort it and recreate the answer.

#### Approach

* Kotlin: withIndex, sortedByDescending.
* Rust: using indices vec and recreating the result makes us use .clone(), so better use zip.

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun sortPeople(names: Array<String>, heights: IntArray) = names
        .withIndex()
        .sortedByDescending { heights[it.index] }
        .map { it.value }

```
```rust

    pub fn sort_people(names: Vec<String>, heights: Vec<i32>) -> Vec<String> {
        let mut zip: Vec<_> = names.into_iter().zip(heights.into_iter()).collect();
        zip.sort_unstable_by_key(|(n, h)| -h);
        zip.into_iter().map(|(n, h)| n).collect()
    }

```

