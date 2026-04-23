---
layout: leetcode-entry
title: "2434. Using a Robot to Print the Lexicographically Smallest String"
permalink: "/leetcode/problem/2025-06-06-2434-using-a-robot-to-print-the-lexicographically-smallest-string/"
leetcode_ui: true
entry_slug: "2025-06-06-2434-using-a-robot-to-print-the-lexicographically-smallest-string"
---

[2434. Using a Robot to Print the Lexicographically Smallest String](https://leetcode.com/problems/using-a-robot-to-print-the-lexicographically-smallest-string/description/) medium
[blog post](https://leetcode.com/problems/using-a-robot-to-print-the-lexicographically-smallest-string/solutions/6816531/kotlin-rust-by-samoylenkodmitry-ymge/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06062025-2434-using-a-robot-to-print?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/M21ti_jTLbA)
![1.webp](/assets/leetcode_daily_images/96440917.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1011

#### Problem TLDR

Smallest string by writing to stack #medium #string

#### Intuition

Kind of solved myself.

Chain-of-thougths:

```j
    // bac
    //       b
    //       ba
    //           ab
    //       c
    //           abc

    // bddac           -> ac ddb
    //      bd
    //          db
    //      da
    //          dbad
    //          dbadc
    //          db ad c  so, we can rotate any part of the string
    //                       but only once, non-intersecting
    //                       and can swap intervals
    //                       how to do it in O(n) and optimally?
    //                       (11 minute)
    //       bdda
    //            a
    //       bddc
    //            ac
    //            acddb

    // bddacab
    //      *  -> abcaddb (wrong), aabcddb is correct, hot to get it?
    //     bdd  a
    //     bddc aa
    //     bddc aab
    //     ok, so it looks like we can skip any chars
    //         and skipped chars would be the prefix (reversed)
    //         (19 minute)
    //  abcabcbcadac
    //                 what the strategy for skipping?
    //                 let's take all 'a''s
    //  bcbcbcdc aaaa -> aaaa cdcbcbcb   looks like it works
    //                                   (21 minute)
    //  ok, it is not working for "bac" -> abc
    //                             we can leave all the suffix
    //  bacaaccccbbb
    //  bc aaa ccccbbb, after the last 'a' we do the same for suffix
    //  cb     cccc bbb
    //         bbb cccc
    //  aaa bbb cccc cb

    // bacaabccbeb -> aaa bbb e ccc b
    // bc aaa bccbeb
    //        cce bbb
    // bc cce
    //    aaa bbb ecc cb   (32 minute)

    // ok, 40 minute, "bac" wrong (acb instead of abc)
    // the strategy was wrong, after 'a' better to take 'b' then 'c'

    // 45 minute, another corner case "vzhofnpo"
    // look for hints
    // hint1: knew
    // hint2: knew
    // hint3: knew, so it is all about the implementation details

    // vzhofnpo    abcdefghijklmnopqrstuvwxyz
    //     *. .         *       ..            vzho    f
    //   *  . .           *     ..
    //      * .                 *.            vzho    fn
    //        *                  *            vzh     fno
    //                                        vzhp    fnoo > fnoh (57 minute)
    //        *                  *            vzh     fno

    // (62 minute) TLE (and I'm happy)

```

Observations:
* take the `smallest` chars first
* take all the `current` chars, skip others adding them to stack, stop at the rightmost
* increment the `current` char
* before going to the right, get all the `smaller` chars from the stack

Another solutions from `u/votrubac/` is counting:
* build the frequencies
* put `every` char to stack as you go
* if all `current chars` are taken, drain all the `smaller` chars from the stack

#### Approach

* strictly synchronize variables, current char and its index
* attention to the order of operations: `before increment`, `increment current`, `after increment`
* we can write `for c in s` or `for c in 'a'..'z'` for the same algorithm

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 68ms
    fun robotWithString(s: String) = buildString {
        val ix = IntArray(26) { -1 }; val t = Stack<Char>()
        var c = 'a'; for (i in s.indices) ix[s[i] - 'a'] = i
        for (i in 0..s.length) {
            while (ix[c - 'a'] < i && ++c <= 'z')
                while (t.size > 0 && t.last() <= c) append(t.pop())
            if (i < s.length) if (s[i] == c) append(c) else t += s[i]
        }
    }

```
```kotlin

// 63ms
    fun robotWithString(s: String) = buildString {
        val idx = IntArray(26) { -1 }; val t = ArrayList<Char>()
        var i = 0; for (i in s.indices) idx[s[i] - 'a'] = i
        for (c in 'a'..'z') {
            while (t.size > 0 && t.last() <= c) append(t.removeLast())
            while (i <= idx[c - 'a']) {
                if (s[i] == c) append(c) else t += s[i]
                i++
            }
        }
        while (i < s.length) t += s[i++]
        for (i in t.lastIndex downTo 0) append(t[i])
    }

```
```kotlin

// 50ms
    fun robotWithString(s: String) = buildString {
        val f = IntArray(26); for (c in s) ++f[c - 'a']
        var j = 0; val t = ArrayList<Char>()
        for (c in s) {
            t += c; --f[c - 'a']
            while (j < 25 && f[j] < 1) ++j
            while (t.size > 0 && t.last() <= 'a' + j) append(t.removeLast())
        }
    }

```
```kotlin

// 44ms
    fun robotWithString(s: String) = buildString {
        val ix = IntArray(27) { -1 }; val t = ArrayList<Char>()
        var c = 'a' - 1; for (i in s.indices) ix[s[i] - 'a'] = i
        var j = -1; ix[26] = s.length
        for ((i, si) in s.withIndex()) {
            while (j < i) {
                c++; j = ix[c - 'a']
                while (t.size > 0 && t.last() <= c) append(t.removeLast())
            }
            if (si == c) append(c) else t += si
        }
        for (i in t.lastIndex downTo 0) append(t[i])
    }

```
```rust

// 9ms
    pub fn robot_with_string(s: String) -> String {
        let mut f = [0; 26]; for b in s.bytes() { f[(b - b'a') as usize] += 1 }
        let (mut j, mut t, mut r) = (0, vec![], String::new());
        for b in s.bytes() {
            t.push(b); f[(b - b'a') as usize] -= 1;
            while j < 25 && f[j] < 1 { j += 1 }
            while t.len() > 0 && t[t.len() - 1] <= b'a' + j as u8 { r.push(t.pop().unwrap() as char) }
        } r
    }

```
```c++

// 48ms
    string robotWithString(string s) {
        int f[26]={}, j = 0; string r, t;for (auto& c: s) ++f[c - 'a'];
        for (auto& c: s) {
            t += c; --f[c - 'a']; while (j < 25 && f[j] < 1) ++j;
            while (size(t) > 0 && t.back() <= 'a' + j) r += t.back(), t.pop_back();
        } return r;
    }

```

