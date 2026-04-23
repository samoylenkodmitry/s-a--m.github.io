---
layout: leetcode-entry
title: "2845. Count of Interesting Subarrays"
permalink: "/leetcode/problem/2025-04-25-2845-count-of-interesting-subarrays/"
leetcode_ui: true
entry_slug: "2025-04-25-2845-count-of-interesting-subarrays"
---

[2845. Count of Interesting Subarrays](https://leetcode.com/problems/count-of-interesting-subarrays/description/) medium
[blog post](https://leetcode.com/problems/count-of-interesting-subarrays/solutions/6685452/kotlin-rust-by-samoylenkodmitry-5517/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25042025-2845-count-of-interesting?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/faStErxWK04)
![1.webp](/assets/leetcode_daily_images/8e036fc2.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/969

#### Problem TLDR

Subarrays, count a[i]%m=k is c%m=k #medium #hash_map

#### Intuition

Failed to solve.

Wrong two-pointers reasoning:

```j

    // 0 1 2 3
    // 3 1 9 6       m=3 k=0
    //               0,3,6,9... cnt indices
    // *   * *       the interesting indices are known
    // what possible patterns are
    // ...***..., ....*....*....*..., .***.***, .***.***.***
    //                                  i..j
    //                                 i.....j
    // window of 0, window of 3, window of 6,...
    // what if k=1, m=3
    // ..*..*..*.
    // window of 1, window of 4, window of x*m+1
    // how many windows possible?
    // m = 1, k = 0, windows_count = n, so it is n^2 algo
    // 012345678
    // .***.***.***.***..
    // .i j. j .j j. j ..
    //   i .j j. j .j j..
    //    i. j .j j.j j..  every j is a valid end
    //                     it is (i+m-1)%m (* only)
    // .***.***.***.***..
    //  i     .   .        s=0 js=0
    //  j i   .   .        s=0 js=1
    //   j  i .   .        s=0 js=2
    //  * j  i.   .        s=1 js=3
    //   *  j i   .        s=1 js=4
    //  * *  j  i .        s=2 js=5
    //   *  * j  i.        s=2 js=6
    //  * * * * j i        the number of prefix stars=4, number of js=7
    //                     s=(js+1)/3 ?
    //     .
    //      \how to count this dot?

```
I've almost come to the conclusion of counting prefix good `j`'s. But do not understood how can I also count `non-divisible` dots.

Then I used the hints:
* if we are at `i` and have good `count[i]`
* then how many `good` starting `j`s are? `j` is good if `(count[i] - count[j]) % m == k`. Number of good starting `j`s is `freq[count[j]]`.
* I have to be fluid with modulo arithmetics: `(a - b) % m == k`, `a%m - b%m == k`, `b%m == a%m - k`, `b%m == (a%m + m - k)%m` (+m to make positive)

#### Approach

* zero case is `map[0] = 1`
* `non-divisible` positions are counted because we increase `r` by previous running sum `cnt`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun countInterestingSubarrays(nums: List<Int>, m: Int, k: Int): Long {
        var r = 0L; var cnt = 0; var map = HashMap<Int, Int>(); map[0] = 1
        for (x in nums) {
            cnt = (cnt + (if (x % m == k) 1 else 0)) % m
            r += map[(cnt + m - k) % m] ?: 0
            map[cnt] = 1 + (map[cnt] ?: 0)
        }
        return r
    }

```
```rust

    pub fn count_interesting_subarrays(n: Vec<i32>, m: i32, k: i32) -> i64 {
        let (mut cnt, mut r, mut map) = (0, 0, HashMap::new()); map.insert(0, 1);
        for x in n {
            if x % m == k { cnt += 1}
            r += map.get(&((cnt + m - k) % m)).unwrap_or(&0);
            *map.entry(cnt % m).or_insert(0) += 1
        } r
    }

```
```c++

    long long countInterestingSubarrays(vector<int>& n, int m, int k) {
        unordered_map<int, int> map; long long r = 0; int cnt = 0; map[0] = 1;
        for (int x: n) {
            cnt = (cnt + (x % m == k)) % m;
            r += map[(cnt + m - k) % m];
            ++map[cnt];
        } return r;
    }

```

