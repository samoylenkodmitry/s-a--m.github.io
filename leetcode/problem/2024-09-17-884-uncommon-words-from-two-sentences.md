---
layout: leetcode-entry
title: "884. Uncommon Words from Two Sentences"
permalink: "/leetcode/problem/2024-09-17-884-uncommon-words-from-two-sentences/"
leetcode_ui: true
entry_slug: "2024-09-17-884-uncommon-words-from-two-sentences"
---

[884. Uncommon Words from Two Sentences](https://leetcode.com/problems/uncommon-words-from-two-sentences/description/) easy
[blog post](https://leetcode.com/problems/uncommon-words-from-two-sentences/solutions/5798216/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17092024-884-uncommon-words-from?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/9u4npHO16fA)
![1.webp](/assets/leetcode_daily_images/d9de55d5.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/738

#### Problem TLDR

Unique words from two strings #easy

#### Intuition

We can count frequencies by using a HashMap

#### Approach

* treat two strings like a single, no difference
* there is a `groupBy` in Kotlin (in Rust it is in external crate itertools)
* c++ has a `stringstream`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun uncommonFromSentences(s1: String, s2: String) =
        "$s1 $s2".split(" ").groupBy { it }
        .filter { (k, v) -> v.size < 2 }.keys.toList()

```
```rust

    pub fn uncommon_from_sentences(s1: String, s2: String) -> Vec<String> {
        let mut freq = HashMap::new();
        for w in s1.split_whitespace() { *freq.entry(w).or_insert(0) += 1 }
        for w in s2.split_whitespace() { *freq.entry(w).or_insert(0) += 1 }
        freq.into_iter().filter(|(k, v)| *v == 1).map(|(k, v)| k.to_string()).collect()
    }

```
```c++

    vector<string> uncommonFromSentences(string s1, string s2) {
        unordered_map<string, int> freq; vector<string> res;
        string s = s1 + " " + s2; stringstream ss(s); string w;
        while (getline(ss, w, ' ')) ++freq[w];
        for (auto kv: freq) if (kv.second == 1) res.push_back(kv.first);
        return res;
    }

```

