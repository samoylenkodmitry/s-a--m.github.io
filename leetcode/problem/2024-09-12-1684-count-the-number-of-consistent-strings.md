---
layout: leetcode-entry
title: "1684. Count the Number of Consistent Strings"
permalink: "/leetcode/problem/2024-09-12-1684-count-the-number-of-consistent-strings/"
leetcode_ui: true
entry_slug: "2024-09-12-1684-count-the-number-of-consistent-strings"
---

[1684. Count the Number of Consistent Strings](https://leetcode.com/problems/count-the-number-of-consistent-strings/description/) easy
[blog post](https://leetcode.com/problems/count-the-number-of-consistent-strings/solutions/5774693/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12092024-1684-count-the-number-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/in268wIDmeg)
![1.webp](/assets/leetcode_daily_images/62e9e1e4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/732

#### Problem TLDR

Count words with `allowed` characters #easy

#### Intuition

There are total of `26` characters, check them.

#### Approach

* we can use a HashSet
* we can use a bit mask

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun countConsistentStrings(allowed: String, words: Array<String>) =
        words.count { it.all { it in allowed }}

```
```rust

    pub fn count_consistent_strings(allowed: String, words: Vec<String>) -> i32 {
        let set: HashSet<_> = allowed.bytes().collect();
        words.iter().filter(|w| w.bytes().all(|b| set.contains(&b))).count() as _
    }

```
```c++

    int countConsistentStrings(string allowed, vector<string>& words) {
        auto bits = [](string w) {
            int mask = 0; for (int i = 0; i < w.length(); i++)
                mask |= 1 << (w[i] - 'a');
            return mask;
        };
        int mask = bits(allowed);
        return std::count_if(words.begin(), words.end(),
            [mask, &bits](string w){return (mask | bits(w)) == mask;});
    }

```

