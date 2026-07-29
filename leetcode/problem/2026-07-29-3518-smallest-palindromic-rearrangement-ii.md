---
layout: leetcode-entry
title: "3518. Smallest Palindromic Rearrangement II"
permalink: "/leetcode/problem/2026-07-29-3518-smallest-palindromic-rearrangement-ii/"
leetcode_ui: true
entry_slug: "2026-07-29-3518-smallest-palindromic-rearrangement-ii"
---

[3518. Smallest Palindromic Rearrangement II](https://leetcode.com/problems/smallest-palindromic-rearrangement-ii/solutions/8427960/kotlin-rust-by-samoylenkodmitry-rw9d/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29072026-3518-smallest-palindromic?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Nguogeufyeo)

https://dmitrysamoylenko.com/leetcode/

![29.07.2026.webp](/assets/leetcode_daily_images/29.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1435

#### Problem TLDR

K-th palindrome rearrangement

#### Intuition

Didn't solved (even after yesterday preparation)
```j
    // i looked at this problem yesterday
    // let's see if i could solve it today...
    //
    // btw yesterday acceptance rate in leetcode was 14%, now it is 30%
    //
    // this is a hard math statistics problem
    // first take the half of the frequencies
    //
    // now we have to construct it letter by letter from 'a' to 'z'
    // to take a letter the count of possible permutations of leftovers f's
    //                  should be less than k
    //
    // now the hard part is to count how many permutations we have
    //
    // p = n!/f!(n-f)!  n - total count, f - frequency we take
    //
    // and we need to take entire leftover frequencies, meaning it is
    // p1*p2*p3
    //
    // n!/f!(n-f)! is calculated with Pascal's Triangle
    //                                new row = previus left + right
    //
    // 26 minute: wrong answer
    // 40 minute: MLE (pascal triangle n^2)
```
* count frequencies - the problem now is the K-th permutation of frequencies (halved)
* construct string letter by letter; consider each letter in 'a'..'z', take the first where the *suffix* permutations count of remaining frequency is bigger or equal than K

#### Approach

* number of permutations of remaining frequency is just taking each letter indifidually n!/f[c]!(n-f[c])! and muliplying, then adjust n -= f[c]
* n!/f!(n-f)! could be computed as res *= n-i+1/i without pascal triangle because we can and should stop when reaching the K

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun smallestPalindrome(s: String, k: Int): String {
        var K = 1L*k; val max = K+1; val f = IntArray(26); for (c in s) ++f[c-'a']
        if (f.count { it%2 > 0} > 1) return ""; val mid = (0..25).firstOrNull { f[it]%2 > 0}
        fun C(n: Int, r: Int) = (1..min(r,n-r)).fold(1L){ a,i -> min(max, a*(n-i+1)/i) }
        fun P() = f.fold(1L to f.sum()) {(res,n), v -> min(max, res*C(n,v)) to  n - v}.first
        for (i in 0..25) f[i] /= 2; if (K > P()) return ""
        val half = buildString {
            for (step in 1..s.length/2) for (c in 0..25) if (f[c]>0) {
                --f[c]; val cnt = P(); if (cnt >= K){append('a'+c);break}; ++f[c]; K -= cnt
            }
        }
        return half + (mid?.let{'a'+it}?:"") + half.reversed()
    }
```
```rust
    pub fn smallest_palindrome(s: String, k: i32) -> String {
        let (mut K, M, mut f, mut h)=(k as i64, k as i64+1, [0i64;26], vec![]);
        for b in s.bytes() { f[(b-97)as usize]+=1}; let mid = (0..26).find(|&i|f[i]%2>0);
        if f.iter().map(|v|v%2).sum::<i64>()>1 { return "".into()}; for v in &mut f {*v/=2}
        let C=|n:i64,r:i64|(1..=r.min(n-r)).fold(1,|a,i|M.min(a*(n-i+1)/i));
        let P=|f:&[i64]|f.iter().fold((1,f.iter().sum()),|(r,n),&v|(M.min(r*C(n,v)),n-v)).0;
        if K > P(&f) { return "".into()}
        for _ in 0..s.len()/2 { for i in 0..26 { if f[i] > 0 {
            f[i]-=1; let cnt = P(&f); if cnt>=K {h.push(97+i as u8);break};f[i]+=1; K-=cnt }}}
        let mut res = h.clone(); if let Some(i) = mid { res.push(97+i as u8)}
        res.extend(h.into_iter().rev()); String::from_utf8(res).unwrap()
    }
```

