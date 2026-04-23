---
layout: leetcode-entry
title: "2559. Count Vowel Strings in Ranges"
permalink: "/leetcode/problem/2025-01-02-2559-count-vowel-strings-in-ranges/"
leetcode_ui: true
entry_slug: "2025-01-02-2559-count-vowel-strings-in-ranges"
---

[2559. Count Vowel Strings in Ranges](https://leetcode.com/problems/count-vowel-strings-in-ranges/description/) medium
[blog post](https://leetcode.com/problems/count-vowel-strings-in-ranges/solutions/6217667/kotlin-rust-by-samoylenkodmitry-mlgr/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02012025-2559-count-vowel-strings?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/rWpq_owJcO0)
[deep-dive](https://notebooklm.google.com/notebook/ea707d2a-7e43-4cdb-94e6-f20276ee82f2/audio)
![1.webp](/assets/leetcode_daily_images/f3c06d75.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/852

#### Problem TLDR

Count words[q[0]..q[1]] starting and ending with "aeiou" #medium

#### Intuition

The prefix sum will answer to each query in O(1) time.

#### Approach

* to check vowels, we can use a HashSet, bitmask or just a String
* in some languages `bool` can be converted to `int`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun vowelStrings(words: Array<String>, queries: Array<IntArray>): List<Int> {
        val fr = IntArray(words.size + 1); val wv = "aeiou"
        for ((i, w) in words.withIndex()) fr[i + 1] = fr[i] +
            if (w[0] in wv && w.last() in wv) 1 else 0
        return queries.map { (f, t) -> fr[t + 1] - fr[f] }
    }

```
```rust

    pub fn vowel_strings(words: Vec<String>, queries: Vec<Vec<i32>>) -> Vec<i32> {
        let mut fr = vec![0; words.len() + 1]; let wv = |b| 1065233 >> (b - b'a') & 1;
        for (i, w) in words.iter().map(|w| w.as_bytes()).enumerate() {
            fr[i + 1] = fr[i] + wv(w[0]) * wv(w[w.len() - 1])
        }
        queries.iter().map(|q| fr[q[1] as usize + 1] - fr[q[0] as usize]).collect()
    }

```
```c++

    vector<int> vowelStrings(vector<string>& words, vector<vector<int>>& queries) {
        unordered_set<char> vw({'a', 'e', 'i', 'o', 'u'}); vector<int> f(1), res;
        for (auto &w: words) f.push_back(f.back() + (vw.count(w.front()) && vw.count(w.back())));
        for (auto &q: queries) res.push_back(f[q[1] + 1] - f[q[0]]);
        return res;
    }

```

