---
layout: leetcode-entry
title: "49. Group Anagrams"
permalink: "/leetcode/problem/2024-02-06-49-group-anagrams/"
leetcode_ui: true
entry_slug: "2024-02-06-49-group-anagrams"
---

[49. Group Anagrams](https://leetcode.com/problems/group-anagrams/description/) medium
[blog post](https://leetcode.com/problems/group-anagrams/solutions/4685010/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06022024-49-group-anagrams?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/kJG2XizPubY)
![image.png](/assets/leetcode_daily_images/dad8801b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/497

#### Problem TLDR

Group words by chars in them.

#### Intuition

We can use char's frequencies or just sorted words as keys to grouping.

#### Approach

Use the standard API for Kotlin and Rust:
* groupBy vs no grouping method in Rust (but have in itertools)
* entry().or_insert_with for Rust
* keys are faster to just sort instead of count in Rust

#### Complexity

- Time complexity:
$$O(mn)$$, for counting, mlog(n) for sorting

- Space complexity:
$$O(mn)$$

#### Code

```kotlin

    fun groupAnagrams(strs: Array<String>): List<List<String>> =
       strs.groupBy { it.groupBy { it } }.values.toList()

```
```rust

  pub fn group_anagrams(strs: Vec<String>) -> Vec<Vec<String>> {
    let mut groups = HashMap::new();
    for s in strs {
      let mut key: Vec<_> = s.bytes().collect();
      key.sort_unstable();
      groups.entry(key).or_insert_with(Vec::new).push(s);
    }
    groups.into_values().collect()
  }

```

