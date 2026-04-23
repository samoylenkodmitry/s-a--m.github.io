---
layout: leetcode-entry
title: "451. Sort Characters By Frequency"
permalink: "/leetcode/problem/2024-02-07-451-sort-characters-by-frequency/"
leetcode_ui: true
entry_slug: "2024-02-07-451-sort-characters-by-frequency"
---

[451. Sort Characters By Frequency](https://leetcode.com/problems/sort-characters-by-frequency/description) medium
[blog post](https://leetcode.com/problems/sort-characters-by-frequency/solutions/4690399/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07022024-451-sort-characters-by-frequency?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/LoTEmZ1Vl7M)
![image.png](/assets/leetcode_daily_images/30601b25.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/498

#### Problem TLDR

Sort string by char's frequencies.

#### Intuition

The optimal solution would be to sort `[128]` size array of frequencies, then build a string in O(n). There are some other ways, however...

#### Approach

Let's explore the shortest versions of code by using the API:
* Kotlin: groupBy, sortedBy, flatMap, joinToString
* Rust: vec![], sort_unstable_by_key, just sorting the whole string takes 3ms

#### Complexity

- Time complexity:
$$O(n)$$, or O(nlog(n)) for sorting the whole string

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun frequencySort(s: String) = s
    .groupBy { it }.values
    .sortedBy { -it.size }
    .flatMap { it }
    .joinToString("")

```
```rust

  pub fn frequency_sort(s: String) -> String {
    let mut f = vec![0; 128];
    for b in s.bytes() { f[b as usize] += 1 }
    let mut cs: Vec<_> = s.chars().collect();
    cs.sort_unstable_by_key(|&c| (-f[c as usize], c));
    cs.iter().collect()
  }

```

