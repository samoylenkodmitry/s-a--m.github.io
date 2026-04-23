---
layout: leetcode-entry
title: "1239. Maximum Length of a Concatenated String with Unique Characters"
permalink: "/leetcode/problem/2024-01-23-1239-maximum-length-of-a-concatenated-string-with-unique-characters/"
leetcode_ui: true
entry_slug: "2024-01-23-1239-maximum-length-of-a-concatenated-string-with-unique-characters"
---

[1239. Maximum Length of a Concatenated String with Unique Characters](https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters/description) medium
[blog post](https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters/solutions/4612267/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23012024-1239-maximum-length-of-a?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/tnX2d8zkPJ0)
![image.png](/assets/leetcode_daily_images/f2199895.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/481

#### Problem TLDR

Max length subsequence of strings array with unique chars.

#### Intuition

Let's do a brute-force Depth-First Search and keep track of used chars so far.

#### Approach

* we must exclude all strings with duplicate chars
* we can use bit masks, then `mask xor word` must not be equal `mask or word` for them not to intersect

#### Complexity

- Time complexity:
$$O(2^n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun maxLength(arr: List<String>): Int {
    val sets = arr.filter { it.toSet().size == it.length }
    fun dfs(i: Int, s: Set<Char>): Int = if (i == sets.size) 0
      else max(
        if (sets[i].any { it in s }) 0 else
        sets[i].length + dfs(i + 1, s + sets[i].toSet()),
        dfs(i + 1, s)
      )
    return dfs(0, setOf())
  }

```
```rust

  pub fn max_length(arr: Vec<String>) -> i32 {
    let bits: Vec<_> = arr.into_iter()
      .filter(|s| s.len() == s.chars().collect::<HashSet<_>>().len())
      .map(|s| s.bytes().fold(0, |m, c| m | 1 << (c - b'a')))
      .collect();
    fn dfs(bits: &[i32], i: usize, mask: i32) -> i32 {
      if i == bits.len() { 0 } else {
      dfs(bits, i + 1, mask).max(
        if (bits[i] | mask != bits[i] ^ mask) { 0 } else
        { bits[i].count_ones() as i32 + dfs(bits, i + 1, mask | bits[i]) }
      )}
    }
    dfs(&bits, 0, 0)
  }

```

