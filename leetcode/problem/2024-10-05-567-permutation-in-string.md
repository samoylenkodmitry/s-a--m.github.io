---
layout: leetcode-entry
title: "567. Permutation in String"
permalink: "/leetcode/problem/2024-10-05-567-permutation-in-string/"
leetcode_ui: true
entry_slug: "2024-10-05-567-permutation-in-string"
---

[567. Permutation in String](https://leetcode.com/problems/permutation-in-string/description/) medium
[blog post](https://leetcode.com/problems/permutation-in-string/solutions/5872376/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05102024-567-permutation-in-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Tw-qYBBot3M)
![1.webp](/assets/leetcode_daily_images/52addce6.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/757

#### Problem TLDR

Is `s2` contains permutation of `s1`? #medium #two_pointers

#### Intuition

Only the characters count matter, so count them with two pointers: one increases the count, the other decreases.

#### Approach

* to avoid all alphabet checks, count frequency intersections with zero

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun checkInclusion(s1: String, s2: String): Boolean {
        val freq = IntArray(26); val target = IntArray(26)
        for (c in s1) target[c - 'a']++; var j = 0
        return s2.any { c ->
            freq[c - 'a']++
            while (freq[c - 'a'] > target[c - 'a']) freq[s2[j++] - 'a']--
            (0..25).all { freq[it] == target[it] }
        }
    }

```
```rust

    pub fn check_inclusion(s1: String, s2: String) -> bool {
        let (mut freq, mut cnt, mut j, s2) = ([0; 26], 0, 0, s2.as_bytes());
        for b in s1.bytes() {
            cnt += (freq[(b - b'a') as usize] == 0) as i32;
            freq[(b - b'a') as usize] += 1
        }
        (0..s2.len()).any(|i| {
            let f = freq[(s2[i] - b'a') as usize];
            freq[(s2[i] - b'a') as usize] -= 1;
            if f == 1 { cnt -= 1 } else if f == 0 { cnt += 1 }
            while freq[(s2[i] - b'a') as usize] < 0 {
                let f = freq[(s2[j] - b'a') as usize];
                freq[(s2[j] - b'a') as usize] += 1;
                if f == -1 { cnt -= 1 } else if f == 0 { cnt += 1 }
                j += 1
            }
            cnt == 0
        })
    }

```
```c++

    bool checkInclusion(string s1, string s2) {
        int f[26], c = 0, j = 0; for (char x: s1) c += !f[x - 'a']++;
        auto adjust = [&](int i, int inc) { return (f[i] += inc) == inc ? 1 : !f[i] ? -1 : 0; };
        return any_of(s2.begin(), s2.end(), [&](char x) {
            c += adjust(x - 'a', -1);
            while (f[x - 'a'] < 0) c += adjust(s2[j++] - 'a', 1);
            return !c;
        });
    }

```

