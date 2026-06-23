---
layout: leetcode-entry
title: "3699. Number of ZigZag Arrays I"
permalink: "/leetcode/problem/2026-06-23-3699-number-of-zigzag-arrays-i/"
leetcode_ui: true
entry_slug: "2026-06-23-3699-number-of-zigzag-arrays-i"
---

[3699. Number of ZigZag Arrays I](https://leetcode.com/problems/number-of-zigzag-arrays-i/solutions/8353350/kotlin-rust-by-samoylenkodmitry-qgz9/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23062026-3699-number-of-zigzag-arrays?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/4ARmo1QHhB8)

https://dmitrysamoylenko.com/leetcode/

![23.06.2026.webp](/assets/leetcode_daily_images/23.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1399

#### Problem TLDR

Ways to make l..r zigzag sequence

#### Intuition

Didn't solve.
```j
    // 15 minute: TLE, O(n^3), hint: prefix sums?
    // f(i, p, 1) = f(i+1, l, 2) + f(i+1, l+1, 2) + ... f(i+1, p-1, 2)

    // f(i, p, 2) = f(i+1, p+1, 1) + f(i+1, p+2, 1)+...+f(i+1,r,1)

    // f(i+1, l, 2) = f(i+2, l+1, 1) + f(i+2, l+2, 1) + ... f(i+2, p-1, 1)

    // f(i+1, p+1, 1) = f(i+2, l, 2) + f(i+2, l+1, 2)+...+f(i+2,p,2)
    // 28 minute: give up
    //
    // f(i,p,1) = sum{k=l..p-1}(f(i+1,k,2))
    // f(i,p-1,1) = sum{k=l..p-2}(f(i+1,k,2))
    // f(i,p,1)-f(i,p-1,1)=f(i+1,p-1,2)
    // f(i,p,1)=f(i,p-1,1)+f(i+1,p-1,2)
    // f(i,p,2)=f(i,p+1,2)+f(i+1,p+1,1)
```
Top down O(n^2) can be derived (see above), but still gives TLE.
Bottom up:
1) move range to 0..r-l
2) use symmetry *2
3) dp[v] is the number of arrays ending with value v
4) dp[v]=sum(dp[0..v]) odd and sum(dp[v..r-l]) even
5) the sum(dp[..]) is just a running sum variable

#### Approach

* not sure if I could solve this or similar next time, the jump to bottom up is not obvious

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun zigZagArrays(n: Int, l: Int, r: Int): Int {
        val r = r-l; val M = 1000000007; val dp = IntArray(r+1){1}
        for (i in 1..<n) {
            var pre = 0
            for (v in if (i%2>0) 0..r else r downTo 0)
                { val pre2 = pre + dp[v]; dp[v] = pre; pre = pre2%M }
        }
        return dp.fold(0){r,t->(r+t)%M}*2%M
    }
```
```rust
    pub fn zig_zag_arrays(n: i32, l: i32, r: i32) -> i32 {
        let r = (r-l+1) as usize; let M = 1000000007; let mut dp = vec![1;r];
        for i in 1..n {
            let mut s = 0;
            if i & 1 > 0 { for v in 0..r { s=(s+replace(&mut dp[v],s))%M} }
            else { for v in (0..r).rev() { s=(s+replace(&mut dp[v],s))%M} }
        }
        (dp.iter().fold(0, |s, &t| (s+t)%M)*2%M) as _
    }
```

