---
layout: leetcode-entry
title: "3042. Count Prefix and Suffix Pairs I"
permalink: "/leetcode/problem/2025-01-08-3042-count-prefix-and-suffix-pairs-i/"
leetcode_ui: true
entry_slug: "2025-01-08-3042-count-prefix-and-suffix-pairs-i"
---

[3042. Count Prefix and Suffix Pairs I](https://leetcode.com/problems/count-prefix-and-suffix-pairs-i/description/) easy
[blog post](https://leetcode.com/problems/count-prefix-and-suffix-pairs-i/solutions/6248509/kotlin-rust-by-samoylenkodmitry-234n/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08012025-3042-count-prefix-and-suffix?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/HPEIOgQBvsE)
[deep-dive](https://notebooklm.google.com/notebook/a0905d0e-44a6-4197-adfa-8dd3d8b4b7b8/audio)
![1.webp](/assets/leetcode_daily_images/05f9701e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/858

#### Problem TLDR

Count prefix-suffix matched pairs #easy

#### Intuition

The brute-force is accepted.

More interesting solutions are:
* Trie: traverse each word forwards and backwards, if suffix trie has the same word as prefix trie, add frequency
* HashMap: just store words in a frequency HashMap and traverse it on each new word
* Robin-Karp rolling hash & KMP/z-function: same idea as with Trie, but check rolling hash to match visited hashes, then do quick-match with KMP/z-function

#### Approach

* on a smaller input the O(n^2) solutions are faster
* we can use a single Trie with the key of `(prefix-letetr, suffix-letter)`

#### Complexity

- Time complexity:
$$O(n^2w^2)$$, or O(nw) for more optimal

- Space complexity:
$$O(1)$$ or O(n)

#### Code

```kotlin

    fun countPrefixSuffixPairs(words: Array<String>) =
        (0..<words.size).flatMap { i -> (i + 1..<words.size).map { i to it }}
        .count { (i, j) -> words[j].startsWith(words[i]) && words[j].endsWith(words[i])}

```
```rust

    pub fn count_prefix_suffix_pairs(words: Vec<String>) -> i32 {
        #[derive(Default)] struct T(usize, i32, HashMap<u8, T>);
        let (mut tf, mut tb, mut res) = (T::default(), T::default(), 0);
        for (p, w) in words.iter().map(|w| w.as_bytes()).enumerate() {
            let (mut f, mut b) = (&mut tf, &mut tb);
            for i in 0..w.len() {
                let (cf, cb) = (w[i], w[w.len() - i - 1]);
                f = f.2.entry(cf).or_default();
                b = b.2.entry(cb).or_default();
                if f.0 > 0 && f.0 == b.0 { res += f.1 }
            }
            f.0 = p + 1; b.0 = p + 1; f.1 += 1; b.1 += 1
        }
        res
    }

```
```c++

    int countPrefixSuffixPairs(vector<string>& words) {
        unordered_map<string, int> m; int res = 0; m[words[0]] = 1;
        for (int i = 1; i < words.size(); ++m[words[i++]])
            for (auto& [prev, freq] : m)
                if (words[i].starts_with(prev) && words[i].ends_with(prev))
                    res += freq;
        return res;
    }

```

