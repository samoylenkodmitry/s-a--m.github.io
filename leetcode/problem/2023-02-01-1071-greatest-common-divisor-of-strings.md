---
layout: leetcode-entry
title: "1071. Greatest Common Divisor of Strings"
permalink: "/leetcode/problem/2023-02-01-1071-greatest-common-divisor-of-strings/"
leetcode_ui: true
entry_slug: "2023-02-01-1071-greatest-common-divisor-of-strings"
---

[1071. Greatest Common Divisor of Strings](https://leetcode.com/problems/greatest-common-divisor-of-strings/description/) easy

[blog post](https://leetcode.com/problems/greatest-common-divisor-of-strings/solutions/3125925/kotlin-gcd/)

```kotlin
    fun gcdOfStrings(str1: String, str2: String): String {
        if (str1 == "" || str2 == "") return ""
        if (str1.length == str2.length) return if (str1 == str2) str1 else ""
        fun gcd(a: Int, b: Int): Int {
            return if (a == 0) b
            else gcd(b % a, a)
        }
        val len = gcd(str1.length, str2.length)
        for (i in 0..str1.lastIndex)  if (str1[i] != str1[i % len]) return ""
        for (i in 0..str2.lastIndex)  if (str2[i] != str1[i % len]) return ""
        return str1.substring(0, len)

    }

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/105
#### Intuition
Consider the following example: `ababab` and `abab`.
If we scan them linearly, we see, the common part is `abab`.
Now, we need to check if the last part from the first `abab_ab` is a part of the common part: `ab` vs `abab`.
This can be done recursively, and we come to the final consideration: `"" vs "ab"`.
That all procedure give us the common divisor - `ab`.
The actual hint is in the method's name ;)

#### Approach
We can first find the length of the greatest common divisor, then just check both strings.

#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(n)$$

