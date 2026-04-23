---
layout: leetcode-entry
title: "2109. Adding Spaces to a String"
permalink: "/leetcode/problem/2024-12-03-2109-adding-spaces-to-a-string/"
leetcode_ui: true
entry_slug: "2024-12-03-2109-adding-spaces-to-a-string"
---

[2109. Adding Spaces to a String](https://leetcode.com/problems/adding-spaces-to-a-string/description/) medium
[blog post](https://leetcode.com/problems/adding-spaces-to-a-string/solutions/6107731/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03122024-2109-adding-spaces-to-a?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Y65Sl49UrMg)
[deep-dive](https://notebooklm.google.com/notebook/f59650ec-6c8e-4f60-8740-5ab5408c974c/audio)
![1.webp](/assets/leetcode_daily_images/dc3e3436.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/821

#### Problem TLDR

Insert spaces into string #medium

#### Intuition

Iterate over string and adjust second pointer for spaces or iterate over spaces and insert substrings.

#### Approach

* Kotlin has a `slice` for strings
* Rust strings can append `&[..]` slices

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun addSpaces(s: String, spaces: IntArray) = buildString {
        for ((j, i) in spaces.withIndex())
              k
        append(s.drop(spaces.last()))
    }

```
```rust

    pub fn add_spaces(s: String, spaces: Vec<i32>) -> String {
        let mut r = String::new();
        for (i, &j) in spaces.iter().enumerate() {
            r += &s[r.len() - i..j as usize]; r += " "
        }; r += &s[*spaces.last().unwrap() as usize..]; r
    }

```
```c++

    string addSpaces(string s, vector<int>& spaces) {
        string r;
        for (int i = 0, j = 0; i < s.size(); ++i)
            j < spaces.size() && i == spaces[j]
                ? j++, r += " ", r += s[i] : r += s[i];
        return r;
    }

```

