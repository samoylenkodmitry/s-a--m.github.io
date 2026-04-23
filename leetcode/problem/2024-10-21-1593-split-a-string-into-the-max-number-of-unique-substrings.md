---
layout: leetcode-entry
title: "1593. Split a String Into the Max Number of Unique Substrings"
permalink: "/leetcode/problem/2024-10-21-1593-split-a-string-into-the-max-number-of-unique-substrings/"
leetcode_ui: true
entry_slug: "2024-10-21-1593-split-a-string-into-the-max-number-of-unique-substrings"
---

[1593. Split a String Into the Max Number of Unique Substrings](https://leetcode.com/problems/split-a-string-into-the-max-number-of-unique-substrings/description/) medium
[blog post](https://leetcode.com/problems/split-a-string-into-the-max-number-of-unique-substrings/solutions/5945933/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21102024-1593-split-a-string-into?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/6jBGCch6c9Q)
[deep-dive](https://notebooklm.google.com/notebook/84e0bcbc-4940-43d1-bf49-a655f9cc4e20/audio)
![1.webp](/assets/leetcode_daily_images/8a7f0a17.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/775

#### Problem TLDR

Max count of unique split parts #medium #backtrack

#### Intuition

The problem size is only `16` length max, so a full Depth-First Search is accepted. Store the current substrings in a HashSet and find a maximum size of it. Iterate on all substrings starting with the current position `i`.

#### Approach

* some code golf possible by reusing the function definition and storing uniqs separately (but it is not the production code)
* in Rust slices also action like a pointer
* notice how `&&` and `,` operator in C++ make the code look clever

#### Complexity

- Time complexity:
$$O(n^n)$$, iterating `n` times on each depth, max depth is `n`

- Space complexity:
$$O(n)$$, for the recursion depth and a HashSet

#### Code

```kotlin

    val uniqs = HashSet<String>()
    fun maxUniqueSplit(s: String): Int =
        (1..s.length).maxOfOrNull { i ->
            if (uniqs.add(s.take(i)))
                1 + maxUniqueSplit(s.drop(i)).also { uniqs -= s.take(i) }
            else 0
        } ?: 0

```
```rust

    pub fn max_unique_split(s: String) -> i32 {
        let (mut res, mut uniqs) = (0, HashSet::new());
        fn dfs(s: &str, res: &mut i32, uniqs: &mut HashSet<String>) {
            *res = uniqs.len().max(*res as usize) as i32;
            for j in 0..s.len() {
                if uniqs.insert(s[..=j].to_string()) {
                    dfs(&s[j + 1..], res, uniqs); uniqs.remove(&s[..=j]);
                }
            }
        }
        dfs(&s, &mut res, &mut uniqs); res
    }

```
```c++

    int maxUniqueSplit(string s) {
        unordered_set<string> uniqs; int res = 0;
        function<void(int)>dfs = [&](int i) {
            res = max(res, (int) uniqs.size());
            for (int j = i; j < s.length(); ++j)
                uniqs.insert(s.substr(i, j - i + 1)).second &&
                    (dfs(j + 1), uniqs.erase(s.substr(i, j - i + 1)));
        }; dfs(0); return res;
    }

```

