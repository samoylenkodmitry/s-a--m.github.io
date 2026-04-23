---
layout: leetcode-entry
title: "2914. Minimum Number of Changes to Make Binary String Beautiful"
permalink: "/leetcode/problem/2024-11-05-2914-minimum-number-of-changes-to-make-binary-string-beautiful/"
leetcode_ui: true
entry_slug: "2024-11-05-2914-minimum-number-of-changes-to-make-binary-string-beautiful"
---

[2914. Minimum Number of Changes to Make Binary String Beautiful](https://leetcode.com/problems/minimum-number-of-changes-to-make-binary-string-beautiful/description/) medium
[blog post](https://leetcode.com/problems/minimum-number-of-changes-to-make-binary-string-beautiful/solutions/6009991/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05112024-2914-minimum-number-of-changes?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/itMLi9ZgkGo)
[deep-dive](https://notebooklm.google.com/notebook/c7bb9b89-571e-4295-8a55-ff43b50b4e1d/audio)
![1.webp](/assets/leetcode_daily_images/e6d5624d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/791

#### Problem TLDR

Min changes to make even-sized `0` and `1` #medium

#### Intuition

Observing some examples:
```j

    // 111011
    // 111111 -> 1
    // 110011 -> 1

```
It is clear that it doesn't matter which bits we change `0`->`1` or `1`->`0`. So, the simplest solution is to just count continuous zeros and ones and greedily fix odds.

Something like this:
```kotlin
        while (++i < s.length) {
            while (i < s.length && s[i] == s[j]) i++
            res += (i - j) % 2
            j = i - (i - j) % 2
        }
```

The cleverer solution comes from the idea: if all substrings are even-sized, they are split at even positions. That means we can scan 2-sized chunks and find all the incorrect splits `c[0] != c[1]`.

#### Approach

* let's do code golf

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minChanges(s: String) =
        s.chunked(2).count { it[0] != it[1] }

```
```rust

    pub fn min_changes(s: String) -> i32 {
        s.as_bytes().chunks(2).map(|c| (c[0] != c[1]) as i32).sum()
    }

```
```c++

    int minChanges(string s) {
        int cnt = 0;
        for (int i = 0; i < s.size(); i += 2)
            cnt += (s[i] ^ s[i + 1]) & 1;
        return cnt;
    }

```

