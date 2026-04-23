---
layout: leetcode-entry
title: "1481. Least Number of Unique Integers after K Removals"
permalink: "/leetcode/problem/2024-02-16-1481-least-number-of-unique-integers-after-k-removals/"
leetcode_ui: true
entry_slug: "2024-02-16-1481-least-number-of-unique-integers-after-k-removals"
---

[1481. Least Number of Unique Integers after K Removals](https://leetcode.com/problems/least-number-of-unique-integers-after-k-removals/description) medium
[blog post](https://leetcode.com/problems/least-number-of-unique-integers-after-k-removals/solutions/4735342/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16022024-1481-least-number-of-unique?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/2wTeTM1rKdY)
![image.png](/assets/leetcode_daily_images/6ddb8663.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/507

#### Problem TLDR

Min uniq count after removing k numbers.

#### Intuition

Just to be sure what the problem is about, let's write some other examples: `[1,2,3,4,4] k = 3`, `[1,2,3,4,4,4] k = 3`, `[1,2,3,3,4,4,4] k = 3`. The first two will give the same unswer `1`, the last one is `2`, however. As soon as we understood  the problem, just implement the algorithm: sort numbers by frequency and remove from smallest to the largest.

#### Approach

Let's try to make the code shorter, by using languages:
* Kotlin: `asList`, `groupingBy`, `eachCount`, `sorted`, `run`
* Rust: `entry+or_insert`, `Vec::from_iter`, `into_values`, `sort_unstable`, `fold`

#### Complexity

- Time complexity:
$$O(nlog(n))$$, worst case, all numbers are uniq

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun findLeastNumOfUniqueInts(arr: IntArray, k: Int) = arr
    .asList().groupingBy { it }.eachCount()
    .values.sorted().run {
      var c = k
      size - count { c >= it.also { c -= it } }
    }

```
```rust

  pub fn find_least_num_of_unique_ints(arr: Vec<i32>, mut k: i32) -> i32 {
    let mut freq = HashMap::new();
    for x in arr { *freq.entry(x).or_insert(0) += 1 }
    let mut freq = Vec::from_iter(freq.into_values());
    freq.sort_unstable();
    freq.iter().fold(freq.len() as i32, |acc, count| {
      k -= count;
      if k < 0 { acc } else { acc - 1 }
    })
  }

```

