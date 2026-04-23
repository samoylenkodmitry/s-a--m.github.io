---
layout: leetcode-entry
title: "1910. Remove All Occurrences of a Substring"
permalink: "/leetcode/problem/2025-02-11-1910-remove-all-occurrences-of-a-substring/"
leetcode_ui: true
entry_slug: "2025-02-11-1910-remove-all-occurrences-of-a-substring"
---

[1910. Remove All Occurrences of a Substring](https://leetcode.com/problems/remove-all-occurrences-of-a-substring/description/) medium
[blog post](https://leetcode.com/problems/remove-all-occurrences-of-a-substring/solutions/6406645/kotlin-rust-by-samoylenkodmitry-ql6u/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11022025-1910-remove-all-occurrences?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/honEq9BhOoQ)
![1.webp](/assets/leetcode_daily_images/45a2ccb8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/892

#### Problem TLDR

Remove substring recursively #medium

#### Intuition

The problem size is 1000, we can use n^2 brute-force.

#### Approach

* the order matters

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun removeOccurrences(s: String, part: String) =
        s.fold("") { r, c -> (r + c).removeSuffix(part) }

```
```rust

    pub fn remove_occurrences(mut s: String, part: String) -> String {
        while let Some(i) = s.find(&part) {
            s.replace_range(i..i + part.len(), "")
        }; s
    }

```
```c++

    string removeOccurrences(string s, string part) {
        while (size(s) > s.find(part)) s.erase(s.find(part), size(part));
        return s;
    }

```

