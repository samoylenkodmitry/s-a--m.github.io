---
layout: leetcode-entry
title: "791. Custom Sort String"
permalink: "/leetcode/problem/2024-03-11-791-custom-sort-string/"
leetcode_ui: true
entry_slug: "2024-03-11-791-custom-sort-string"
---

[791. Custom Sort String](https://leetcode.com/problems/custom-sort-string/description/) medium
[blog post](https://leetcode.com/problems/custom-sort-string/solutions/4857722/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11032024-791-custom-sort-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/9aFn8ccDZuI)
![2024-03-11_09-08.jpg](/assets/leetcode_daily_images/026f2a07.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/535

#### Problem TLDR

Construct string from `s` using `order` #medium

#### Intuition

Two ways to solve: use sort (we need a stable sort algorithm), or use frequency.

#### Approach

When using sort, take care of `-1` case.
When using frequency, we can use it as a counter too (` -= 1`).

#### Complexity

- Time complexity:
$$O(n)$$, or nlog(n) for sorting

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun customSortString(order: String, s: String) = s
    .toMutableList()
    .sortedBy { order.indexOf(it).takeIf { it >= 0 } ?: 200 }
    .joinToString("")

```
```rust

  pub fn custom_sort_string(order: String, s: String) -> String {
    let (mut freq, mut res) = (vec![0; 26], String::new());
    for b in s.bytes() { freq[(b - b'a') as usize] += 1 }
    for b in order.bytes() {
      let i = (b - b'a') as usize;
      while freq[i] > 0 {  freq[i] -= 1; res.push(b as char) }
    }
    for b in s.bytes() {
      if freq[(b - b'a') as usize] > 0 { res.push(b as char) }
    }; res
  }

```

