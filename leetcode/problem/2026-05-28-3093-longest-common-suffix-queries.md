---
layout: leetcode-entry
title: "3093. Longest Common Suffix Queries"
permalink: "/leetcode/problem/2026-05-28-3093-longest-common-suffix-queries/"
leetcode_ui: true
entry_slug: "2026-05-28-3093-longest-common-suffix-queries"
---

[3093. Longest Common Suffix Queries](https://leetcode.com/problems/longest-common-suffix-queries/solutions/8298645/kotlin-rust-by-samoylenkodmitry-4tj7/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28052026-3093-longest-common-suffix?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/AiPXs-7kVkI)

https://dmitrysamoylenko.com/leetcode/

![28.05.2026.webp](/assets/leetcode_daily_images/28.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1373

#### Problem TLDR

Query common suffixes match

#### Intuition

Use Trie.
Update the shortest index as you go.

#### Approach

* in Rust the interesting pattern is an arena allocation

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun stringIndices(c: Array<String>, q: Array<String>) = run {
        class T(var i: Int): HashMap<Char,T>()
        val r = T(c.indexOf(c.minBy{it.length}))
        for ((i,w) in c.withIndex()) { var t = r
            for (j in w.lastIndex downTo 0) t = t.getOrPut(w[j]){T(i)}
                .also { if (w.length < c[it.i].length) it.i = i }
        }
        q.map { var t = r; for (c in it.reversed()) t = t[c] ?: break; t.i }
    }
```
```rust
    pub fn string_indices(c: Vec<String>, q: Vec<String>) -> Vec<i32> {
        let mut n = vec![([0; 26], (0..c.len()).min_by_key(|&i| c[i].len()).unwrap())];
        for (i, w) in c.iter().enumerate() { let mut u = 0;
            for b in w.bytes().rev() { let k = (b - b'a') as usize;
                if n[u].0[k] == 0 { n[u].0[k] = n.len(); n.push(([0; 26], i)) }
                u = n[u].0[k]; if w.len() < c[n[u].1].len() { n[u].1 = i }
            }
        }
        q.iter().map(|w| { let mut u = 0;
            for b in w.bytes().rev() {
                let v = n[u].0[(b - b'a') as usize]; if v == 0 { break } u = v
            } n[u].1 as _
        }).collect()
    }
```

