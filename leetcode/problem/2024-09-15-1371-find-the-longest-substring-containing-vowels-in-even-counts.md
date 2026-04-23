---
layout: leetcode-entry
title: "1371. Find the Longest Substring Containing Vowels in Even Counts"
permalink: "/leetcode/problem/2024-09-15-1371-find-the-longest-substring-containing-vowels-in-even-counts/"
leetcode_ui: true
entry_slug: "2024-09-15-1371-find-the-longest-substring-containing-vowels-in-even-counts"
---

[1371. Find the Longest Substring Containing Vowels in Even Counts](https://leetcode.com/problems/find-the-longest-substring-containing-vowels-in-even-counts/description/) medium
[blog post](https://leetcode.com/problems/find-the-longest-substring-containing-vowels-in-even-counts/solutions/5789593/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15092024-1371-find-the-longest-substring?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ngmRfRO2vuo)
![1.webp](/assets/leetcode_daily_images/e5dd61c6.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/736

#### Problem TLDR

Longest substring with even number of "aeiou" #medium #bit_manipulation #two_pointers

#### Intuition

Can't solve it without the hint.

The hint is: use a bit mask for vowels.

Now, let's observe how we can do this:

```j

    // hello
    // hell
    //    ^ xor(hell) == xor(he)
    // helolelo

```

The bit mask for `hell` is equal to the bit mask of `he` - both contains a single `e`. So, we can store the first encounter of each uniq bit mask to compute the difference between them: `hell - he = ll (our result)`.

#### Approach

* we can use a HashMap or just an array for the `bits` and for the `indices`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$, `2^(vowels.size)` for `indices`, `vowels.size` for `bits`

#### Code

```kotlin

    fun findTheLongestSubstring(s: String): Int {
        val freqToInd = mutableMapOf<Int, Int>()
        val bit = mapOf('a' to 1, 'e' to 2, 'i' to 4, 'o' to 8, 'u' to 16)
        var freq = 0; freqToInd[0] = -1
        return s.indices.maxOf { i ->
            freq = freq xor (bit[s[i]] ?: 0)
            i - freqToInd.getOrPut(freq) { i }
        }
    }

```
```rust

    pub fn find_the_longest_substring(s: String) -> i32 {
        let (mut freq, mut freq_to_ind) = (0, [s.len(); 32]); freq_to_ind[0] = 0;
        let bit = [1, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0];
        s.bytes().enumerate().map(|(i, b)| {
            freq ^= bit[(b - b'a') as usize]; freq_to_ind[freq] = freq_to_ind[freq].min(i + 1);
            i - freq_to_ind[freq] + 1
        }).max().unwrap() as _
    }

```
```c++

    int findTheLongestSubstring(string s) {
        int freq = 0, res = 0;
        unordered_map<int, int> freqToInd = { {0, -1} };
        for (auto i = 0; i < s.length(); i++) {
            freq ^= (1 << (string("aeiou").find(s[i]) + 1)) >> 1;
            if (!freqToInd.count(freq)) freqToInd[freq] = i;
            res = max(res, i - freqToInd[freq]);
        }
        return res;
    }

```

