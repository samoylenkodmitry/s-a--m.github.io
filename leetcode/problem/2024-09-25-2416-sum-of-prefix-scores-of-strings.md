---
layout: leetcode-entry
title: "2416. Sum of Prefix Scores of Strings"
permalink: "/leetcode/problem/2024-09-25-2416-sum-of-prefix-scores-of-strings/"
leetcode_ui: true
entry_slug: "2024-09-25-2416-sum-of-prefix-scores-of-strings"
---

[2416. Sum of Prefix Scores of Strings](https://leetcode.com/problems/sum-of-prefix-scores-of-strings/description/) hard
[blog post](https://leetcode.com/problems/sum-of-prefix-scores-of-strings/solutions/5831195/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25092024-2416-sum-of-prefix-scores?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/OnVfpPBdvTg)
![1.webp](/assets/leetcode_daily_images/8e00013a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/746

#### Problem TLDR

Counts of words with same prefixes #hard #trie

#### Intuition

The HashMap counter gives OOM. There is also a Trie data structure for prefixes problems.

#### Approach

* To avoid `Option<Box>` in Rust we can implement Trie as just a pointers to a `Vec` positions, where the actual data lies. (the time drops from 213ms to 145ms)

#### Complexity

- Time complexity:
$$O(nw)$$

- Space complexity:
$$O(w)$$

#### Code

```kotlin

    class Trie(var freq: Int = 0) : HashMap<Char, Trie>()
    fun sumPrefixScores(words: Array<String>) = Trie().run {
        for (w in words) {
            var t = this
            for (c in w) t = t.getOrPut(c) { Trie() }.apply { freq++ }
        }
        words.map { var t = this; it.sumOf { t = t[it]!!; t.freq } }
    }

```
```rust

    pub fn sum_prefix_scores(words: Vec<String>) -> Vec<i32> {
        #[derive(Clone, Default)] struct Trie((i32, [Option<Box<Trie>>; 26]));
        let (mut root, a) = (Trie::default(), b'a' as usize);
        for w in words.iter() { let mut t = &mut root; for b in w.bytes() {
            t = t.0.1[b as usize - a].get_or_insert_with(|| Box::new(Trie::default()));
            t.0.0 += 1
        }}
        words.iter().map(|w| { let mut t = &root;
            w.bytes().map(|b| { t = t.0.1[b as usize - a].as_ref().unwrap(); t.0.0 }).sum()
        }).collect()
    }

```
```rust

    pub fn sum_prefix_scores(words: Vec<String>) -> Vec<i32> {
        #[derive(Clone, Default)] struct Trie((i32, [usize; 26]));
        let (mut nodes, a) = (vec![Trie::default()], b'a' as usize);
        for w in words.iter() { let mut t = 0; for b in w.bytes() {
            if nodes[t].0.1[b as usize - a] == 0 {
                nodes[t].0.1[b as usize - a] = nodes.len(); nodes.push(Trie::default())
            }
            t = nodes[t].0.1[b as usize - a]; nodes[t].0.0 += 1
        }}
        words.iter().map(|w| { let mut t = &nodes[0];
            w.bytes().map(|b| { t = &nodes[t.0.1[b as usize - a]]; t.0.0 }).sum()
        }).collect()
    }

```
```c++

    vector<int> sumPrefixScores(vector<string>& words) {
        struct Trie { int c{}; array<Trie*, 26> k{}; };
        Trie root;
        for (auto& w : words) { Trie* t = &root;
            for (char c : w)
                ++(t = t->k[c-97] ? t->k[c-97] : (t->k[c-97] = new Trie))->c;
        }
        std::vector<int> res;
        for (auto& w : words) { Trie* t = &root; int freq = 0;
            for (char c : w) freq += (t = t->k[c-97])->c;
            res.push_back(freq);
        }
        return res;
    }

```

