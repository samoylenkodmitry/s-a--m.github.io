---
layout: leetcode-entry
title: "1207. Unique Number of Occurrences"
permalink: "/leetcode/problem/2024-01-17-1207-unique-number-of-occurrences/"
leetcode_ui: true
entry_slug: "2024-01-17-1207-unique-number-of-occurrences"
---

[1207. Unique Number of Occurrences](https://leetcode.com/problems/unique-number-of-occurrences/) easy
[blog post](https://leetcode.com/problems/unique-number-of-occurrences/solutions/4579328/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17012024-1207-unique-number-of-occurrences?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/qMvrHh2kJ9U)
![image.png](/assets/leetcode_daily_images/4d496d8c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/474

#### Problem TLDR

Are array frequencies unique.

#### Intuition

Just count frequencies.

#### Approach

Let's use some Kotlin's API:

* asList
* groupingBy
* eachCount
* groupBy
* run

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun uniqueOccurrences(arr: IntArray) =
      arr.asList().groupingBy { it }.eachCount().values.run {
        toSet().size == size
      }

```

```rust

  pub fn unique_occurrences(arr: Vec<i32>) -> bool {
    let occ = arr.iter().fold(HashMap::new(), |mut m, &x| {
      *m.entry(x).or_insert(0) += 1; m
    });
    occ.len() == occ.values().collect::<HashSet<_>>().len()
  }

```

