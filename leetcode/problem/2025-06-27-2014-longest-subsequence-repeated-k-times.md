---
layout: leetcode-entry
title: "2014. Longest Subsequence Repeated k Times"
permalink: "/leetcode/problem/2025-06-27-2014-longest-subsequence-repeated-k-times/"
leetcode_ui: true
entry_slug: "2025-06-27-2014-longest-subsequence-repeated-k-times"
---

[2014. Longest Subsequence Repeated k Times](https://leetcode.com/problems/longest-subsequence-repeated-k-times/description/) hard
[blog post](https://leetcode.com/problems/longest-subsequence-repeated-k-times/solutions/6890919/kotlin-rust-by-samoylenkodmitry-t5aq/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27062025-2014-longest-subsequence?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/cqLEjrAWcIQ)
![1.webp](/assets/leetcode_daily_images/752b0ed5.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1032

#### Problem TLDR

Longest k-repeating subsequence #hard #backtracking

#### Intuition

Used the hints.
1. find all `good` characters (at least `k` frequent)
2. do DFS with backtracking
3. prune by only taking at most `n/k` chars, each frequency at most `f[c] / k`

#### Approach

* great speed up: `don't build subsequence further if current is not k-repeating` (300ms to 90ms)
* another way is BFS, explore new length layer by layer (Rust code, no pruning optimizations)

#### Complexity

- Time complexity:
$$O(n^r)$$ `r = max_freq[a..z] / k`

- Space complexity:
$$O(r)$$ recursion depth

#### Code

```kotlin

// 91ms
    fun longestSubsequenceRepeatedK(s: String, k: Int): String {
        var res = ""; val f = IntArray(26); for (c in s) ++f[c - 'a']
        val cnt = IntArray(26); val seq = CharArray(s.length / k); var sz = 0
        fun check() {
            var i = 0; var fr = if (sz > 0) 0 else k
            if (sz > 0) for (c in s) if (c == seq[i % sz]) if (++i % sz == 0) fr++
            if (fr < k) return
            if (sz > res.length) res = String(seq, 0, sz)
            for (c in 25 downTo 0) if (f[c] >= k && cnt[c] < f[c] / k) {
                ++cnt[c]; seq[sz++] = 'a' + c
                check()
                --cnt[c]; --sz
            }
        }
        check()
        return res
    }

```
```rust

// 416ms
    pub fn longest_subsequence_repeated_k(s: String, k: i32) -> String {
        let (mut q, mut q1, mut res) = (vec![String::from("")], vec![], "".into());
        while q.len() > 0 {
            for sub in &q {
                for c in 'a'..='z' {
                    let next = format!("{}{}", sub.clone(), c);
                    let mut i = 0; let mut r = next.len() * (k as usize);
                    for c in s.bytes() {
                        if c == next.as_bytes()[i % next.len()] {
                            i += 1; if i == r { break }}}
                    if i == r { res = next.clone(); q1.push(next) }
                }}
            (q, q1) = (q1, q); q1.clear()
        } res
    }

```
```c++

// 92ms
    string longestSubsequenceRepeatedK(string s, int k) {
        int f[26] = {}, c[26] = {}; for (char x : s) f[x - 'a']++;
        string res, seq;
        auto dfs = [&](this const auto& dfs) {
            int sz = seq.size(), i = 0, cnt = sz ? 0 : k;
            if (sz)
                for (char x : s)
                    if (x == seq[i % sz] && ++i % sz == 0)
                        if (++cnt == k) break;
            if (cnt < k) return;
            if (sz > (int)res.size()) res = seq;
            for (int x = 25; x >= 0; x--) {
                if (f[x] >= k && c[x] < f[x] / k) {
                    c[x]++; seq.push_back('a' + x);
                    dfs();
                    seq.pop_back(); c[x]--;
                }
            }
        };
        dfs();
        return res;
    }

```

