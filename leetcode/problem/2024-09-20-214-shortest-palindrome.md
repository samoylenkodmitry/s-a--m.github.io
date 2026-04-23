---
layout: leetcode-entry
title: "214. Shortest Palindrome"
permalink: "/leetcode/problem/2024-09-20-214-shortest-palindrome/"
leetcode_ui: true
entry_slug: "2024-09-20-214-shortest-palindrome"
---

[214. Shortest Palindrome](https://leetcode.com/problems/shortest-palindrome/description/) hard
[blog post](https://leetcode.com/problems/shortest-palindrome/solutions/5811452/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20092024-214-shortest-palindrome?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/rorbCSPcMPA)
![1.webp](/assets/leetcode_daily_images/1ffc62ee.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/741

#### Problem TLDR

Prepend to make the shortest palindrome #hard #rolling_hash #knuth-morris-pratt

#### Intuition

The brute-force solution is accepted, so just check all the prefixes.

One optimization is to use a `rolling-hash`: compute it for the prefix and it's reverse. The worst case is still O(n^2) for `aaaaa`-like strings if not skip the `equals` check.

```j

    //  aacecaaa -> aacecaa + a
    //  a        -> a
    //  aa       -> aa
    //  aac      -> caa
    //  aace     -> ecaa
    //  aacec    -> cecaa
    //  aaceca   -> acecaa
    //  aacecaa  -> aacecaa

    //  abc
    //  h(ab) + c = h(h(a) + b) + c = 31*(31*a+b) + c = 31^2a + 31b + c
    //  h(bc) = 31b + c
    //  a + h(bc) = 31^2a + 31b + c

```

The optimal solutino is based on `Knuth-Morris-Pratt` substring search: make an array where each value is the length between suffix and preffix up to current position `ps[i] = max_len_of(s[..i] == s[0..])`. It is cleverly constructed and must be learned beforehand.

Now, to apply to current problem, make a string `s#rev_s`, and find the last value of `ps` - it will tell the maximum length of `s[..] == rev_s[..end]`. For example `abab` and `baba` have the common part of `aba`, and this is what we need to know to make a shortest palindrome: `b + aba_b`:

```j

    //      012345678
    //      abab#baba  p
    // 0    i
    //      j          0
    // 1     i
    //      j          00
    // 2      i
    //       j         001
    // 3       i
    //        j        0012
    // 4        i
    //      j          00120
    // 5         i
    //      j          001200
    // 6          i
    //       j         0012001
    // 7           i
    //        j        00120012
    // 8            i
    //         j       001200123

```

#### Approach

* KMP https://cp-algorithms.com/string/prefix-function.html
* rolling-hash is also useful, we construct it for `f = 31 * f + x` for appending and `f = 31^p + f` for prepending

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun shortestPalindrome(s: String): String {
        var hash1 = 0; var hash2 = 0; var p = 1
        return s.drop(s.withIndex().maxOfOrNull { (i, c) ->
            hash1 = 31 * hash1 + c.code
            hash2 += p * c.code
            p *= 31
            if (hash1 == hash2) i + 1 else 0
        } ?: 0).reversed() + s
    }

```
```rust

    pub fn shortest_palindrome(s: String) -> String {
        let mut rev: Vec<_> = s.bytes().rev().collect();
        let sr = [s.as_bytes(), &[b'#'], &rev[..]].concat();
        let (mut j, mut ps, mut common) = (0, vec![0; sr.len()], 0);
        for i in 1..sr.len() {
            while j > 0 && sr[i] != sr[j] { j = ps[j - 1] }
            if sr[i] == sr[j] { j += 1 }
            ps[i] = j; common = j
        }
        from_utf8(&rev[..s.len() - common]).unwrap().to_owned() + &s
    }

```
```c++

    string shortestPalindrome(string s) {
        string rev = s; std::reverse(rev.begin(), rev.end());
        string sr = s + "#" + rev; int j = 0;
        vector<int> ps(sr.size());
        for (int i = 1; i < sr.size(); i++) {
            while (j > 0 && sr[i] != sr[j]) j = ps[j - 1];
            if (sr[i] == sr[j]) j++;
            ps[i] = j;
        }
        return rev.substr(0, s.size() - ps.back()) + s;
    }

```
```kotlin(space-optimmized-kmp)

    fun shortestPalindrome(s: String): String {
        val f = IntArray(s.length + 1); var j = 0
        for (i in 2..s.length) {
            j = f[i - 1]
            while (j > 0 && s[j] != s[i - 1]) j = f[j]
            if (s[j] == s[i - 1]) f[i] = j + 1
        }
        j = 0
        for (i in s.indices) {
            while (j > 0 && s[j] != s[s.length - i - 1]) j = f[j]
            if (s[j] == s[s.length - i - 1]) j++
        }
        return s.drop(j).reversed() + s
    }

```

