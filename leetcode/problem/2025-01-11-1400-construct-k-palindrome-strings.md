---
layout: leetcode-entry
title: "1400. Construct K Palindrome Strings"
permalink: "/leetcode/problem/2025-01-11-1400-construct-k-palindrome-strings/"
leetcode_ui: true
entry_slug: "2025-01-11-1400-construct-k-palindrome-strings"
---

[1400. Construct K Palindrome Strings](https://leetcode.com/problems/construct-k-palindrome-strings/description/) medium
[blog post](https://leetcode.com/problems/construct-k-palindrome-strings/solutions/6263844/kotlin-rust-by-samoylenkodmitry-kb8f/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11012025-1400-construct-k-palindrome?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/62kjsnc9uZc)
[deep-dive](https://notebooklm.google.com/notebook/620e2a5f-0f4f-4ea5-af26-e90fc88d6c8e/audio)
![1.webp](/assets/leetcode_daily_images/6c5ed528.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/861

#### Problem TLDR

Can make `k` palindromes from string? #medium

#### Intuition

The main difficulty is to define how chars frequencies can be used to make `k` palindromes:
* chars number must be at least `k`, this is a lower boundary
* the `odd` frequencies count must be `<= k`, this is a higher boundary
* `even` frequencies are all dissolved into any number of palindromes

#### Approach

* to find those rules on the fly, we should do attempts on examples (by running the code, or writing them down, or imagine if you can)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$, O(n) for Kotlin golf

#### Code

```kotlin

    fun canConstruct(s: String, k: Int) =
        k <= s.length && s.groupBy { it }
            .values.sumBy { it.size % 2 } <= k

```
```rust

    pub fn can_construct(s: String, k: i32) -> bool {
        let (mut f, k) = (vec![0; 26], k as usize);
        for b in s.bytes() { f[(b - b'a') as usize] += 1 }
        k <= s.len() &&
          (0..26).map(|b| f[b] % 2).sum::<usize>() <= k
    }

```
```c++

    bool canConstruct(string s, int k) {
        int f[26] = {0}, c = 0;
        for (int i = 0; i < s.size(); ++i) c += 2 * (++f[s[i] - 'a'] % 2) - 1;
        return k <= s.size() && c <= k;
    }

```

