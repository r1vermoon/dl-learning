class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        l1=[]
        d1={}
        for i in strs:
            nlist=list(i)
            plist=sorted(nlist)
            newstr=''.join(plist)
            if newstr not in d1:
                d1[newstr]=[i]
            else:
                d1[newstr].append(i) 
        for i in d1:
            print(i,' ',d1[i])
            l1.append(d1[i])  
        return l1
    

test=Solution()
test.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"])

