---
layout: leetcode-entry
title: "2825. Make String a Subsequence Using Cyclic Increments"
permalink: "/leetcode/problem/2024-12-04-2825-make-string-a-subsequence-using-cyclic-increments/"
leetcode_ui: true
entry_slug: "2024-12-04-2825-make-string-a-subsequence-using-cyclic-increments"
---

[2825. Make String a Subsequence Using Cyclic Increments](https://leetcode.com/problems/make-string-a-subsequence-using-cyclic-increments/description/) medium
[blog post](https://leetcode.com/problems/make-string-a-subsequence-using-cyclic-increments/solutions/6111889/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04122024-2825-make-string-a-subsequence?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/IWi-SSbskMM)
[deep-dive](https://notebooklm.google.com/notebook/da901144-a315-4bdf-bbd0-95b83fb535f3/audio)
![1.webp](/assets/leetcode_daily_images/a4c06afc.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/822

#### Problem TLDR

Increase some chars once to make a subsequence #medium

#### Intuition

Attention to the description:
* subsequence vs substring
* rotation at most once
* any positions

Let's scan over `str2` (resulting subsequence) and greedily find positions in `str1` for each of its letters. Compare the char and its rolled `down` version.

#### Approach

* trick from Lee: `(s2[i] - s1[j]) <= 1` (with % 26 added for 'a'-'z' case)

#### Complexity

- Time complexity:
$$O(n + m)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun canMakeSubsequence(str1: String, str2: String): Boolean {
        var j = 0; var i = 0
        while (i < str2.length && j < str1.length)
        if ((str2[i] - str1[j] + 26) % 26 <= 1) { ++i; ++j } else ++j
        return i == str2.length
    }

```
```rust

    pub fn can_make_subsequence(str1: String, str2: String) -> bool {
       k
        while i < s2.len() && j < s1.len() {
            if (s2[i] - s1[j] + 26) % 26 <= 1 { i += 1; j += 1 } else { j += 1 }
        }; i == s2.len()
    }

```
```c++

    bool canMakeSubsequence(string s1, string s2) {
        int i = 0;
        for (int j = 0; j < s1.size() && i < s2.size(); ++j)
            if ((s2[i] - s1[j] + 26) % 26 <= 1) ++i;
        return i == s2.size();
    }

```

