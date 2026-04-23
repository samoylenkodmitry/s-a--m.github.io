---
layout: leetcode-entry
title: "1358. Number of Substrings Containing All Three Characters"
permalink: "/leetcode/problem/2025-03-11-1358-number-of-substrings-containing-all-three-characters/"
leetcode_ui: true
entry_slug: "2025-03-11-1358-number-of-substrings-containing-all-three-characters"
---

[1358. Number of Substrings Containing All Three Characters](https://leetcode.com/problems/number-of-substrings-containing-all-three-characters/description/) medium
[blog post](https://leetcode.com/problems/number-of-substrings-containing-all-three-characters/solutions/6523186/kotlin-rust-by-samoylenkodmitry-j4bf/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11032025-1358-number-of-substrings?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/U61TmhZDhro)
![1.webp](/assets/leetcode_daily_images/cbe41ce5.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/922

#### Problem TLDR

Substrings with [abc] #medium #two_pointers

#### Intuition

First idea: always move the right pointer, and move the left pointer while it is possible to have all [abc]. Add running sum of the prefix length: `aaaaabc` have prefix length `4`, increasing count by 4.

The second order insight is we actually only care about the minimum recent visited index to find the prefix length.

#### Approach

* implement your own idea
* look at others ideas
* implement them
* golf all the solutions

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun numberOfSubstrings(s: String) = IntArray(3).run {
        s.indices.sumOf { set(s[it] - 'a', it + 1); min() }
    }

```
```kotlin(first_idea)

    fun numberOfSubstrings(s: String): Int {
        var j = 0; val f = IntArray(3)
        return s.indices.sumOf { i ->
            f[s[i] - 'a']++ < 1
            while (f[s[j] - 'a'] > 1) f[s[j++] - 'a']--
            if (f.all { it > 0 }) j + 1 else 0
        }
    }

```
```rust

    pub fn number_of_substrings(s: String) -> i32 {
        let mut j = vec![0; 3]; s.bytes().enumerate()
            .map(|(i, b)| { j[(b - b'a') as usize] = i + 1;
                j[0].min(j[1]).min(j[2]) as i32 }).sum::<i32>()
    }

```
```c++

    int numberOfSubstrings(string s) {
        int j[3] = {}, r = 0;
        for (int i = 0; i < size(s); ++i)
            j[s[i] - 'a'] = i + 1, r += min({j[0], j[1], j[2]});
        return r;
    }

```

