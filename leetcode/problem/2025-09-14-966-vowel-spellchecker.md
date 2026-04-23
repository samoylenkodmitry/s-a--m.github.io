---
layout: leetcode-entry
title: "966. Vowel Spellchecker"
permalink: "/leetcode/problem/2025-09-14-966-vowel-spellchecker/"
leetcode_ui: true
entry_slug: "2025-09-14-966-vowel-spellchecker"
---

[966. Vowel Spellchecker](https://leetcode.com/problems/vowel-spellchecker/description) medium
[blog post](https://leetcode.com/problems/vowel-spellchecker/solutions/7188776/kotlin-rust-by-samoylenkodmitry-huhi/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14092025-966-vowel-spellchecker?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/HL2vb7coFHc)

![1.webp](/assets/leetcode_daily_images/2be9ef00.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1112

#### Problem TLDR

Spellcheck with priority: original, case, vowels #medium #regex

#### Intuition

Understand the priority:
1. Exact match
2. Ignore-case match
3. Vowel-as-wildcard match

#### Approach

* do full-search for vowels (7 symbols max, 5 wovels = 7^5) or precompute a wildcards

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 73ms
    fun spellchecker(wl: Array<String>, q: Array<String>): List<String> {
        val orig = wl.groupBy { it }; val lower = wl.groupBy { it.lowercase() }
        val vowel = wl.groupBy { Regex("[eiou]").replace(it.lowercase(), "a")}
        return q.map { q ->
            orig[q]?.first() ?: lower[q.lowercase()]?.first() ?:
            vowel[Regex("[eiou]").replace(q.lowercase(), "a")]?.first() ?: ""
        }
    }

```
```rust

// 13ms
    pub fn spellchecker(w: Vec<String>, q: Vec<String>) -> Vec<String> {
        use itertools::Itertools;
        let exact: HashMap<_,_> = w.iter().map(|s| (s.as_str(), s.as_str())).collect();
        let lower: HashMap<_,_> = w.iter().rev().map(|s| (s.to_lowercase(), s.as_str())).collect();
        let vowel: HashMap<_,_> = w.iter().rev().map(|s| (s.to_lowercase().replace(|c|"eiou".contains(c),"a"), s.as_str())).collect();
        q.iter().map(|s| {
            exact.get(s.as_str()).copied()
            .or(lower.get(&s.to_lowercase()).copied())
            .or(vowel.get(&s.to_lowercase().replace(|c|"eiou".contains(c),"a")).copied())
            .unwrap_or("").into()
        }).collect()
    }

```

